"""Most-popular baseline, optionally time-decayed.

Scores are ``log1p`` of the (decayed) number of users per item and are identical for every user; the
model is the floor every other recommender must beat and the fallback for cold users in serving.
"""

from __future__ import annotations

from typing import Self

import numpy as np

from curate.data.interactions import Interactions
from curate.models.base import FitContext

SECONDS_PER_DAY = 86_400.0


class PopularityRecommender:
    """Rank items by how many users interacted with them in training."""

    name = "popularity"

    def __init__(self, half_life_days: float | None = None) -> None:
        self.half_life_days = half_life_days
        self.item_scores: np.ndarray | None = None

    def fit(self, train: Interactions, context: FitContext) -> Self:  # noqa: ARG002
        """Count users per item, weighting each event by an exponential decay when a half-life is set."""
        if self.half_life_days is None:
            counts = train.item_counts()
        else:
            age_days = (train.timestamps.max() - train.timestamps) / SECONDS_PER_DAY
            decay = np.power(0.5, age_days / self.half_life_days)
            counts = np.bincount(train.items, weights=decay, minlength=train.n_items)
        self.item_scores = np.log1p(counts.astype(np.float64))
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        """The same popularity vector for every requested user."""
        if self.item_scores is None:
            raise RuntimeError("fit must be called before score")
        return np.tile(self.item_scores, (len(users), 1))
