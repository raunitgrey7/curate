"""The recommender protocol shared by every model, and the top-k helper that applies it.

Every model scores all items for a batch of users; ranking, masking of already-seen items and the
evaluation loop live outside the models so they are implemented once and tested once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np
import scipy.sparse as sp

from curate.data.interactions import Catalog, Interactions, UserTable


@dataclass(frozen=True)
class FitContext:
    """Side information available at fit time; feature matrices come from the feature store."""

    catalog: Catalog
    users: UserTable
    item_features: np.ndarray
    item_feature_names: list[str]
    user_features: np.ndarray
    user_feature_names: list[str]


class Recommender(Protocol):
    """A model that learns from interactions and scores every item for given users."""

    name: str

    def fit(self, train: Interactions, context: FitContext) -> Self:
        """Learn from training interactions."""
        ...

    def score(self, users: np.ndarray) -> np.ndarray:
        """Dense ``(len(users), n_items)`` scores; larger is better, ``-inf`` means never recommend."""
        ...


def topk_from_scores(scores: np.ndarray, k: int, exclude: sp.csr_matrix | None = None) -> np.ndarray:
    """Top-k item indices per row, with excluded (seen) items masked out first.

    Ties are broken by item index so the result is deterministic for identical scores.
    """
    if k <= 0 or k > scores.shape[1]:
        raise ValueError("k must be in [1, n_items]")
    masked = np.array(scores, dtype=np.float64, copy=True)
    if exclude is not None:
        coo = exclude.tocoo()
        masked[coo.row, coo.col] = -np.inf
    tiny = -np.arange(masked.shape[1], dtype=np.float64) * 1e-12
    masked = masked + tiny
    part = np.argpartition(-masked, k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(masked, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)
