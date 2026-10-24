"""Pydantic request and response models for the API.

Every request model forbids unknown fields and bounds every number so malformed input is rejected with a
422 before it reaches a model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ModelName = Literal["two_stage", "itemknn", "bpr", "two_tower", "popularity"]
MAX_K = 100


class RecommendRequest(BaseModel):
    """Body of ``POST /v1/recommend``."""

    model_config = ConfigDict(extra="forbid")

    user_id: int = Field(ge=0, description="External user id as in the dataset")
    k: int = Field(default=10, ge=1, le=MAX_K)
    model: ModelName = "two_stage"
    exclude_seen: bool = True
    explain: bool = True


class ExplanationOut(BaseModel):
    """Why one item was recommended."""

    kind: str
    text: str
    anchor_item_id: int | None = None
    anchor_title: str | None = None
    shared_genres: list[str] = Field(default_factory=list)
    support: float | None = None
    similarity: float | None = None


class ItemOut(BaseModel):
    """Catalogue entry."""

    item_id: int
    title: str
    genres: list[str]
    year: int | None = None


class RecommendedItem(ItemOut):
    """A ranked recommendation with its score and explanation."""

    rank: int
    score: float
    explanation: ExplanationOut | None = None


class RecommendResponse(BaseModel):
    """Response of the recommend endpoints."""

    user_id: int
    model: str
    k: int
    fallback: Literal["none", "popularity_cold_start"]
    items: list[RecommendedItem]
    model_latency_ms: float
    request_id: str


class SimilarItem(ItemOut):
    """A neighbour with its similarity."""

    similarity: float


class SimilarResponse(BaseModel):
    """Response of ``GET /v1/items/{item_id}/similar``."""

    item: ItemOut
    neighbours: list[SimilarItem]
    request_id: str


class HistoryResponse(BaseModel):
    """Response of ``GET /v1/users/{user_id}/history``."""

    user_id: int
    n_interactions: int
    items: list[ItemOut]
    request_id: str


class HealthResponse(BaseModel):
    """Response of ``GET /health``."""

    status: Literal["ok"]
    version: str
    dataset: str
    models: list[str]
    n_users: int
    n_items: int
    uptime_seconds: float


class ModelsResponse(BaseModel):
    """Response of ``GET /v1/models``."""

    models: list[str]
    default: str
    train_seconds: dict[str, float]


class ErrorResponse(BaseModel):
    """Error envelope used by every non-2xx response."""

    detail: str
    request_id: str | None = None
