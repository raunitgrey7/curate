"""Offline top-k evaluation of a recommender on a time-based split.

Evaluable users have at least one training interaction and at least one test item they had not seen
before the test window. Items seen in train or validation are masked before the top-k is taken, so the
model is judged only on new items, the way a production system would recommend.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from curate.data.split import Split
from curate.evaluation.metrics import item_popularity, ranking_metrics
from curate.models.base import Recommender, topk_from_scores


@dataclass(frozen=True)
class EvaluationTarget:
    """Everything the evaluator needs about a split, computed once and shared across models."""

    users: np.ndarray
    relevant: sp.csr_matrix
    seen: sp.csr_matrix
    popularity: np.ndarray
    n_items: int

    @classmethod
    def from_split(cls, split: Split) -> EvaluationTarget:
        """Derive evaluable users, the relevance matrix and the exclusion mask from a split."""
        seen = split.seen_before_test()
        test = split.test.to_csr()
        relevant = test - test.multiply(seen)
        relevant.eliminate_zeros()
        has_train = np.asarray(split.train.to_csr().sum(axis=1)).ravel() > 0
        has_relevant = np.asarray(relevant.sum(axis=1)).ravel() > 0
        users = np.flatnonzero(has_train & has_relevant)
        return cls(users, relevant.tocsr(), seen, item_popularity(split.train.to_csr()), split.train.n_items)


@dataclass
class OfflineResult:
    """Metrics for one model on one split."""

    model: str
    n_eval_users: int
    fit_seconds: float
    score_seconds: float
    metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | int | str]:
        """Flat dictionary for JSON output."""
        return {
            "model": self.model,
            "n_eval_users": self.n_eval_users,
            "fit_seconds": round(self.fit_seconds, 3),
            "score_seconds": round(self.score_seconds, 3),
            **{k: round(v, 5) for k, v in self.metrics.items()},
        }


def recommend_for_users(
    model: Recommender, target: EvaluationTarget, k: int, batch_size: int = 512
) -> np.ndarray:
    """Top-k unseen items for every evaluable user, scored in batches."""
    chunks = []
    for start in range(0, len(target.users), batch_size):
        batch = target.users[start : start + batch_size]
        scores = model.score(batch)
        chunks.append(topk_from_scores(scores, k, exclude=target.seen[batch]))
    return np.vstack(chunks) if chunks else np.empty((0, k), dtype=np.int64)


def evaluate_model(
    model: Recommender, target: EvaluationTarget, ks: Iterable[int] = (10, 20), fit_seconds: float = 0.0
) -> OfflineResult:
    """Compute every metric in ``ranking_metrics`` for each list size."""
    ks = sorted(ks)
    start = time.perf_counter()
    recommended = recommend_for_users(model, target, max(ks))
    score_seconds = time.perf_counter() - start
    relevant = target.relevant[target.users]
    metrics: dict[str, float] = {}
    for k in ks:
        metrics.update(ranking_metrics(recommended[:, :k], relevant, target.popularity, target.n_items))
    return OfflineResult(model.name, len(target.users), fit_seconds, score_seconds, metrics)
