"""FastAPI application factory.

``create_app`` wires the middleware (request ids, JSON access logs, metrics, API key auth), the Prometheus
endpoint and the versioned recommendation routes around a ``ServingState``. The state can be injected, which
is how tests and the latency harness run the real app in-process without touching ``artifacts/``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from curate import __version__
from curate.config import Settings
from curate.logging_utils import configure_logging
from curate.serving.metrics import Metrics
from curate.serving.middleware import ApiKeyMiddleware, RequestContextMiddleware
from curate.serving.schemas import (
    MAX_K,
    ErrorResponse,
    HealthResponse,
    HistoryResponse,
    ModelName,
    ModelsResponse,
    RecommendRequest,
    RecommendResponse,
    SimilarResponse,
)
from curate.serving.state import DEFAULT_MODEL, ServingState, UnknownItemError

STATIC_DIR = Path(__file__).resolve().parent / "static"
logger = logging.getLogger(__name__)
ERRORS = {404: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}}


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", ""))


def _state(request: Request) -> ServingState:
    state = getattr(request.app.state, "serving", None)
    if state is None:
        raise HTTPException(status_code=503, detail="model bundle not loaded")
    return state


def _recommend(request: Request, body: RecommendRequest) -> RecommendResponse:
    state = _state(request)
    metrics: Metrics = request.app.state.metrics
    result = state.recommend(body.user_id, body.k, body.model, body.exclude_seen, body.explain)
    metrics.recommendations_total.labels(result["model"]).inc()
    metrics.model_latency.labels(result["model"]).observe(result["model_latency_ms"] / 1000)
    if result["fallback"] != "none":
        metrics.cold_starts_total.inc()
    return RecommendResponse(**result, request_id=_request_id(request))


def create_app(settings: Settings | None = None, state: ServingState | None = None) -> FastAPI:
    """Build the application; the bundle is loaded at startup unless ``state`` is given."""
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    metrics = Metrics.create()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.serving = state or ServingState.load(settings)
        metrics.info.labels(app.state.serving.bundle.dataset_name, __version__).set(1)
        logger.info("bundle loaded", extra=app.state.serving.health())
        yield

    app = FastAPI(title="curate", version=__version__, lifespan=lifespan, responses=ERRORS)
    app.state.metrics = metrics
    app.state.settings = settings
    _register_routes(app)
    app.add_middleware(ApiKeyMiddleware, api_key=settings.api_key)
    app.add_middleware(RequestContextMiddleware, metrics=metrics)
    return app


def _register_routes(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        detail = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        return JSONResponse(status_code=422, content={"detail": detail, "request_id": _request_id(request)})

    @app.exception_handler(HTTPException)
    async def http_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code, content={"detail": exc.detail, "request_id": _request_id(request)}
        )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health(request: Request) -> HealthResponse:
        return HealthResponse(**_state(request).health())

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint(request: Request) -> Response:
        return Response(generate_latest(request.app.state.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", include_in_schema=False)
    async def index() -> Response:
        page = STATIC_DIR / "index.html"
        if page.exists():
            return FileResponse(page)
        return JSONResponse({"service": "curate", "version": __version__, "docs": "/docs"})

    @app.get("/v1/models", response_model=ModelsResponse, tags=["recommendations"])
    async def models(request: Request) -> ModelsResponse:
        bundle = _state(request).bundle
        return ModelsResponse(
            models=sorted(bundle.models), default=DEFAULT_MODEL, train_seconds=bundle.train_seconds
        )

    @app.post("/v1/recommend", response_model=RecommendResponse, tags=["recommendations"])
    async def recommend(request: Request, body: RecommendRequest) -> RecommendResponse:
        return _recommend(request, body)

    @app.get("/v1/recommend/{user_id}", response_model=RecommendResponse, tags=["recommendations"])
    async def recommend_get(
        request: Request,
        user_id: Annotated[int, PathParam(ge=0)],
        k: Annotated[int, Query(ge=1, le=MAX_K)] = 10,
        model: ModelName = DEFAULT_MODEL,
        explain: bool = True,
    ) -> RecommendResponse:
        return _recommend(request, RecommendRequest(user_id=user_id, k=k, model=model, explain=explain))

    @app.get("/v1/items/{item_id}", tags=["catalogue"])
    async def item(request: Request, item_id: int) -> dict[str, object]:
        state = _state(request)
        try:
            return state.item_payload(state.resolve_item(item_id))
        except UnknownItemError as exc:
            raise HTTPException(status_code=404, detail=f"unknown item {item_id}") from exc

    @app.get("/v1/items/{item_id}/similar", response_model=SimilarResponse, tags=["catalogue"])
    async def similar(
        request: Request, item_id: int, k: Annotated[int, Query(ge=1, le=MAX_K)] = 10
    ) -> SimilarResponse:
        try:
            return SimilarResponse(**_state(request).similar(item_id, k), request_id=_request_id(request))
        except UnknownItemError as exc:
            raise HTTPException(status_code=404, detail=f"unknown item {item_id}") from exc

    @app.get("/v1/users/{user_id}/history", response_model=HistoryResponse, tags=["catalogue"])
    async def history(
        request: Request, user_id: int, limit: Annotated[int, Query(ge=1, le=1000)] = 50
    ) -> HistoryResponse:
        result = _state(request).history(user_id, limit)
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown user {user_id}")
        return HistoryResponse(**result, request_id=_request_id(request))

    app.state.started = time.time()
