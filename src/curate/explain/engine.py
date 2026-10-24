"""Deterministic "because you watched" explanations grounded in training co-occurrence.

For a recommended item we look for the history item with the strongest item-kNN similarity to it and
report the measured share of that anchor's audience that also consumed the recommendation, plus the
genres the two share. When item-kNN has no link we fall back to the nearest history item in the
two-tower embedding space, and for users without history we state the popularity count. Every number in
an explanation is read from the fitted models, never invented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from curate.features.store import FeatureStore
from curate.models.itemknn import ItemKNNRecommender
from curate.models.two_tower import TwoTowerRecommender

VERBS = {"movielens_100k": "watched", "retail_synthetic": "bought"}


@dataclass(frozen=True)
class Explanation:
    """One explanation; ``kind`` tells the client which template produced ``text``."""

    kind: str
    text: str
    anchor_item_id: int | None = None
    anchor_title: str | None = None
    shared_genres: tuple[str, ...] = ()
    support: float | None = None
    similarity: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly mapping (tuples become lists)."""
        payload = asdict(self)
        payload["shared_genres"] = list(self.shared_genres)
        return payload


class Explainer:
    """Build explanations from the item-kNN model, the two-tower embeddings and the feature store."""

    def __init__(
        self,
        itemknn: ItemKNNRecommender,
        two_tower: TwoTowerRecommender | None,
        history: sp.csr_matrix,
        store: FeatureStore,
        dataset_name: str,
    ) -> None:
        if itemknn.similarity is None or itemknn.cooccurrence is None or itemknn.item_counts is None:
            raise ValueError("item-kNN model must be fitted")
        self.itemknn = itemknn
        self.two_tower = two_tower
        self.history = history
        self.store = store
        self.verb = VERBS.get(dataset_name, "chose")

    def _meta(self, item: int) -> dict[str, Any]:
        return self.store.get_metadata("item", item) or {
            "item_id": item,
            "title": f"item {item}",
            "genres": [],
        }

    def explain(self, user: int | None, item: int) -> Explanation:
        """Explain why ``item`` is recommended to ``user`` (``None`` for anonymous users)."""
        history = self.history[user].indices if user is not None else np.empty(0, dtype=np.int64)
        history = history[history != item]
        if len(history) == 0:
            return self._popular(item)
        anchor = self._knn_anchor(history, item)
        if anchor is not None:
            return anchor
        embedded = self._embedding_anchor(history, item)
        return embedded if embedded is not None else self._popular(item)

    def _knn_anchor(self, history: np.ndarray, item: int) -> Explanation | None:
        assert self.itemknn.similarity is not None
        assert self.itemknn.cooccurrence is not None
        assert self.itemknn.item_counts is not None
        sims = np.asarray(self.itemknn.similarity[history, item].todense()).ravel()
        best = int(np.argmax(sims))
        if sims[best] <= 0:
            return None
        anchor = int(history[best])
        co = float(self.itemknn.cooccurrence[anchor, item])
        support = (
            co / float(self.itemknn.item_counts[anchor]) if self.itemknn.item_counts[anchor] > 0 else 0.0
        )
        anchor_meta, item_meta = self._meta(anchor), self._meta(item)
        shared = tuple(g for g in anchor_meta.get("genres", []) if g in set(item_meta.get("genres", [])))
        text = f'Because you {self.verb} "{anchor_meta["title"]}": {support:.0%} of the people who {self.verb} it also {self.verb} this one'
        if shared:
            text += f"; both are {', '.join(shared)}"
        return Explanation(
            "because_you_watched",
            text + ".",
            anchor_meta["item_id"],
            anchor_meta["title"],
            shared,
            support,
            float(sims[best]),
        )

    def _embedding_anchor(self, history: np.ndarray, item: int) -> Explanation | None:
        if self.two_tower is None or self.two_tower.item_embeddings is None:
            return None
        embeddings = self.two_tower.item_embeddings
        cosines = embeddings[history] @ embeddings[item]
        best = int(np.argmax(cosines))
        anchor_meta = self._meta(int(history[best]))
        item_meta = self._meta(item)
        shared = tuple(g for g in anchor_meta.get("genres", []) if g in set(item_meta.get("genres", [])))
        text = f'Because you {self.verb} "{anchor_meta["title"]}", which has a similar taste profile (cosine {cosines[best]:.2f}).'
        return Explanation(
            "similar_taste",
            text,
            anchor_meta["item_id"],
            anchor_meta["title"],
            shared,
            None,
            float(cosines[best]),
        )

    def _popular(self, item: int) -> Explanation:
        assert self.itemknn.item_counts is not None
        count = int(self.itemknn.item_counts[item])
        return Explanation("popular", f"Popular: {count} people {self.verb} this.", support=None)
