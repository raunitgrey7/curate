"""ASGI middleware: request ids, structured access logs, Prometheus counters and API key auth.

Implemented as plain ASGI callables rather than ``BaseHTTPMiddleware`` so streaming and background tasks
behave normally and the overhead per request stays in the microseconds.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from curate.logging_utils import request_id_var
from curate.serving.metrics import Metrics

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

REQUEST_ID_HEADER = b"x-request-id"
API_KEY_HEADER = b"x-api-key"
AUTH_EXEMPT = frozenset({"/health", "/metrics", "/", "/favicon.ico"})
logger = logging.getLogger("curate.access")


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key == name:
            return value.decode("latin-1")
    return None


def _route_template(scope: Scope) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else scope.get("path", "unknown")


class RequestContextMiddleware:
    """Assign a request id, time the request, log one JSON line and record metrics."""

    def __init__(self, app: ASGIApp, metrics: Metrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Wrap one HTTP request; other scope types pass straight through."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = _header(scope, REQUEST_ID_HEADER) or uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_var.set(request_id)
        status_holder = {"status": 500}
        started = time.perf_counter()

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER, request_id.encode("latin-1")))
                headers.append(
                    (b"x-process-time-ms", f"{(time.perf_counter() - started) * 1000:.3f}".encode())
                )
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - started
            route = _route_template(scope)
            method = scope.get("method", "GET")
            self.metrics.requests_total.labels(method, route, str(status_holder["status"])).inc()
            self.metrics.request_duration.labels(method, route).observe(elapsed)
            logger.info(
                "request",
                extra={
                    "method": method,
                    "path": scope.get("path"),
                    "status": status_holder["status"],
                    "duration_ms": round(elapsed * 1000, 3),
                },
            )
            request_id_var.reset(token)


class ApiKeyMiddleware:
    """Reject requests without the configured ``X-API-Key`` on every route except health and metrics."""

    def __init__(self, app: ASGIApp, api_key: str | None) -> None:
        self.app = app
        self.api_key = api_key

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Return 401 JSON when the key is missing or wrong."""
        if scope["type"] != "http" or self.api_key is None or scope.get("path") in AUTH_EXEMPT:
            await self.app(scope, receive, send)
            return
        if scope.get("path", "").startswith("/static/") or _header(scope, API_KEY_HEADER) == self.api_key:
            await self.app(scope, receive, send)
            return
        request_id = scope.get("state", {}).get("request_id")
        body = f'{{"detail":"invalid or missing API key","request_id":"{request_id}"}}'.encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
