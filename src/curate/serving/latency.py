"""Measure end-to-end HTTP latency of the recommendation endpoint against a live uvicorn server.

The server runs in a background thread on a free local port and a real HTTP client issues the requests,
so the reported percentiles include serialisation, middleware and the network stack, not only model time.
Users are sampled with a fixed seed so the run is reproducible.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI


@dataclass(frozen=True)
class LatencySummary:
    """Percentiles in milliseconds for one model plus throughput under concurrency."""

    model: str
    n_requests: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    requests_per_second_c8: float

    def as_dict(self) -> dict[str, float | int | str]:
        """Flat mapping for JSON output."""
        return {
            "model": self.model,
            "n_requests": self.n_requests,
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "requests_per_second_c8": round(self.requests_per_second_c8, 1),
        }


def free_port() -> int:
    """Ask the OS for an unused TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class BackgroundServer:
    """Run uvicorn in a thread; use as a context manager."""

    def __init__(self, app: FastAPI, port: int | None = None) -> None:
        self.port = port or free_port()
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning", access_log=False)
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        """Base URL of the running server."""
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> BackgroundServer:
        self.thread.start()
        deadline = time.time() + 30
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("server did not start in time")
            time.sleep(0.02)
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def _percentiles(samples_ms: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(np.percentile(samples_ms, 50)),
        float(np.percentile(samples_ms, 95)),
        float(np.percentile(samples_ms, 99)),
        float(samples_ms.mean()),
    )


def measure_model(
    base_url: str,
    model: str,
    user_ids: np.ndarray,
    k: int = 10,
    warmup: int = 20,
    concurrency: int = 8,
    headers: dict[str, str] | None = None,
) -> LatencySummary:
    """Sequential latency percentiles and throughput at fixed concurrency for one model."""
    with httpx.Client(base_url=base_url, timeout=30.0, headers=headers) as client:
        payloads = [{"user_id": int(u), "k": k, "model": model} for u in user_ids]
        for payload in payloads[:warmup]:
            client.post("/v1/recommend", json=payload).raise_for_status()
        samples = np.empty(len(payloads))
        for i, payload in enumerate(payloads):
            start = time.perf_counter_ns()
            client.post("/v1/recommend", json=payload).raise_for_status()
            samples[i] = (time.perf_counter_ns() - start) / 1e6

    def one(payload: dict[str, int | str]) -> None:
        with httpx.Client(base_url=base_url, timeout=30.0, headers=headers) as c:
            c.post("/v1/recommend", json=payload).raise_for_status()

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(one, payloads))
    throughput = len(payloads) / (time.perf_counter() - started)
    p50, p95, p99, mean = _percentiles(samples)
    return LatencySummary(model, len(payloads), p50, p95, p99, mean, throughput)


def measure_latency(
    app: FastAPI, user_ids: np.ndarray, models: list[str], n_requests: int = 200, seed: int = 42
) -> list[LatencySummary]:
    """Start the app, measure every model with the same sampled users, stop the app."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    rng = np.random.default_rng(seed)
    sample = rng.choice(user_ids, size=n_requests, replace=True)
    api_key = getattr(app.state.settings, "api_key", None)
    headers = {"X-API-Key": api_key} if api_key else None
    with BackgroundServer(app) as server:
        return [measure_model(server.url, model, sample, headers=headers) for model in models]
