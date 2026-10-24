"""Request-time state: the loaded bundle, the feature store, the explainer and the id mappings.

The state object does the work the endpoints expose, so it can be tested without HTTP and the endpoints
stay thin. All scoring is a dense pass over the catalogue for one user, which is a few hundred
microseconds at this catalogue size; a real deployment with millions of items would swap in an ANN index
behind the same interface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from curate.config import Settings
from curate.explain.engine import Explainer
from curate.explain.llm import reword_explanation
from curate.features.store import FeatureStore
from curate.models.base import topk_from_scores
from curate.models.itemknn import ItemKNNRecommender
from curate.models.two_tower import TwoTowerRecommender
from curate.pipeline import STORE_FILE, ServingBundle

DEFAULT_MODEL = "two_stage"


class UnknownItemError(KeyError):
    """Raised for an item id that is not in the catalogue."""


@dataclass
class ServingState:
    """Everything the endpoints need, built once at startup."""

    bundle: ServingBundle
    store: FeatureStore
    explainer: Explainer
    settings: Settings
    started_at: float

    @classmethod
    def load(cls, settings: Settings) -> ServingState:
        """Load the bundle and feature store from ``settings.artifact_dir``."""
        bundle = ServingBundle.load(settings.artifact_dir)
        store = FeatureStore(settings.artifact_dir / STORE_FILE)
        return cls.from_bundle(bundle, store, settings)

    @classmethod
    def from_bundle(cls, bundle: ServingBundle, store: FeatureStore, settings: Settings) -> ServingState:
        """Build the state around an in-memory bundle (used by tests and the latency harness)."""
        itemknn = bundle.models["itemknn"]
        two_tower = bundle.models.get("two_tower")
        if not isinstance(itemknn, ItemKNNRecommender):
            raise TypeError("bundle must contain a fitted item-kNN model under 'itemknn'")
        explainer = Explainer(
            itemknn,
            two_tower if isinstance(two_tower, TwoTowerRecommender) else None,
            bundle.history,
            store,
            bundle.dataset_name,
        )
        return cls(bundle, store, explainer, settings, time.time())

    @property
    def user_index(self) -> dict[int, int]:
        """External user id to row index (cached on first use)."""
        if not hasattr(self, "_user_index"):
            self._user_index = {int(u): i for i, u in enumerate(self.bundle.user_ids)}
        return self._user_index

    @property
    def item_index(self) -> dict[int, int]:
        """External item id to column index (cached on first use)."""
        if not hasattr(self, "_item_index"):
            self._item_index = {int(v): i for i, v in enumerate(self.bundle.item_ids)}
        return self._item_index

    def item_payload(self, item: int) -> dict[str, Any]:
        """Catalogue metadata for an internal item index."""
        meta = self.store.get_metadata("item", item) or {}
        return {
            "item_id": int(meta.get("item_id", self.bundle.item_ids[item])),
            "title": str(meta.get("title", f"item {item}")),
            "genres": list(meta.get("genres", [])),
            "year": int(meta["year"]) if meta.get("year") else None,
        }

    def resolve_item(self, item_id: int) -> int:
        """Internal index for an external item id."""
        try:
            return self.item_index[int(item_id)]
        except KeyError as exc:
            raise UnknownItemError(item_id) from exc

    def recommend(
        self,
        user_id: int,
        k: int,
        model: str = DEFAULT_MODEL,
        exclude_seen: bool = True,
        explain: bool = True,
    ) -> dict[str, Any]:
        """Top-k items for a user; unknown users fall back to popularity."""
        if model not in self.bundle.models:
            raise KeyError(model)
        user = self.user_index.get(int(user_id))
        fallback = "none"
        if user is None:
            model, fallback = "popularity", "popularity_cold_start"
        started = time.perf_counter()
        users = np.asarray([user if user is not None else 0])
        scores = self.bundle.models[model].score(users)
        exclude = self.bundle.history[users] if (exclude_seen and user is not None) else None
        top = topk_from_scores(scores, k, exclude=exclude)[0]
        latency_ms = (time.perf_counter() - started) * 1000
        items = []
        for rank, item in enumerate(top, start=1):
            payload = {**self.item_payload(int(item)), "rank": rank, "score": float(scores[0, item])}
            if explain:
                explanation = self.explainer.explain(user, int(item))
                text = reword_explanation(explanation.text, self.settings)
                payload["explanation"] = {**explanation.as_dict(), "text": text}
            items.append(payload)
        return {
            "user_id": int(user_id),
            "model": model,
            "k": k,
            "fallback": fallback,
            "items": items,
            "model_latency_ms": round(latency_ms, 3),
        }

    def similar(self, item_id: int, k: int) -> dict[str, Any]:
        """Item-kNN neighbours of one catalogue item."""
        item = self.resolve_item(item_id)
        itemknn = self.bundle.models["itemknn"]
        assert isinstance(itemknn, ItemKNNRecommender)
        neighbours, sims = itemknn.similar_items(item, k)
        return {
            "item": self.item_payload(item),
            "neighbours": [
                {**self.item_payload(int(j)), "similarity": float(s)}
                for j, s in zip(neighbours, sims, strict=True)
            ],
        }

    def history(self, user_id: int, limit: int) -> dict[str, Any] | None:
        """Items a known user interacted with, or ``None`` for unknown users."""
        user = self.user_index.get(int(user_id))
        if user is None:
            return None
        items = self.bundle.history[user].indices
        return {
            "user_id": int(user_id),
            "n_interactions": len(items),
            "items": [self.item_payload(int(i)) for i in items[:limit]],
        }

    def health(self) -> dict[str, Any]:
        """Payload for ``/health``."""
        return {
            "status": "ok",
            "version": self.bundle.version,
            "dataset": self.bundle.dataset_name,
            "models": sorted(self.bundle.models),
            "n_users": int(self.bundle.history.shape[0]),
            "n_items": self.bundle.n_items,
            "uptime_seconds": round(time.time() - self.started_at, 3),
        }
