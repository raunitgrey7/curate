"""Top-k ranking metrics plus catalogue coverage and novelty.

All functions take ``recommended`` as an integer array of shape ``(n_users, k)`` and ``relevant`` as a
binary sparse matrix ``(n_users, n_items)``; a hit is a recommended item that is relevant for that user.
Definitions follow the usual conventions and are spelled out in ``docs/EVALUATION.md``:

* Recall@k = hits / number of relevant items (not capped at k, so a user with 30 relevant items cannot
  reach 1.0 at k = 10; this makes the number conservative and comparable across users).
* NDCG@k uses binary gains, ``1 / log2(rank + 1)`` discounts and an ideal DCG over ``min(k, relevant)``.
* Coverage@k is the share of the catalogue that appears in at least one user's list.
* Novelty@k is the mean self-information ``-log2 p(i)`` of recommended items, where ``p(i)`` is the share
  of training users who interacted with ``i`` (Zhou et al. 2010; Vargas and Castells 2011).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def hit_matrix(recommended: np.ndarray, relevant: sp.csr_matrix) -> np.ndarray:
    """Boolean ``(n_users, k)`` matrix marking recommended items that are relevant."""
    if recommended.ndim != 2:
        raise ValueError("recommended must be 2-D (n_users, k)")
    rows = np.repeat(np.arange(recommended.shape[0]), recommended.shape[1])
    hits = np.asarray(relevant[rows, recommended.ravel()]).ravel() > 0
    return hits.reshape(recommended.shape)


def _relevant_counts(relevant: sp.csr_matrix) -> np.ndarray:
    return np.asarray((relevant > 0).sum(axis=1)).ravel().astype(np.float64)


def recall_at_k(recommended: np.ndarray, relevant: sp.csr_matrix) -> float:
    """Mean over users of hits divided by the number of relevant items; users without relevant items are skipped."""
    hits = hit_matrix(recommended, relevant).sum(axis=1)
    counts = _relevant_counts(relevant)
    mask = counts > 0
    return float(np.mean(hits[mask] / counts[mask])) if mask.any() else 0.0


def precision_at_k(recommended: np.ndarray, relevant: sp.csr_matrix) -> float:
    """Mean share of the list that is relevant."""
    return float(hit_matrix(recommended, relevant).mean()) if recommended.size else 0.0


def hit_rate_at_k(recommended: np.ndarray, relevant: sp.csr_matrix) -> float:
    """Share of users with at least one hit in their list."""
    return float(hit_matrix(recommended, relevant).any(axis=1).mean()) if recommended.size else 0.0


def mrr_at_k(recommended: np.ndarray, relevant: sp.csr_matrix) -> float:
    """Mean reciprocal rank of the first hit (zero when the list has no hit)."""
    hits = hit_matrix(recommended, relevant)
    first = np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, 0)
    return float(np.mean(np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0))) if recommended.size else 0.0


def ndcg_at_k(recommended: np.ndarray, relevant: sp.csr_matrix) -> float:
    """Normalised discounted cumulative gain with binary gains."""
    hits = hit_matrix(recommended, relevant).astype(np.float64)
    k = recommended.shape[1]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = hits @ discounts
    counts = np.minimum(_relevant_counts(relevant), k).astype(np.int64)
    ideal = np.cumsum(discounts)
    idcg = np.where(counts > 0, ideal[np.maximum(counts, 1) - 1], 0.0)
    mask = counts > 0
    return float(np.mean(dcg[mask] / idcg[mask])) if mask.any() else 0.0


def coverage_at_k(recommended: np.ndarray, n_items: int) -> float:
    """Share of the catalogue recommended to at least one user."""
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    return float(len(np.unique(recommended)) / n_items)


def item_popularity(train: sp.csr_matrix) -> np.ndarray:
    """Share of training users who interacted with each item, smoothed so no item has zero probability."""
    n_users = train.shape[0]
    counts = np.asarray((train > 0).sum(axis=0)).ravel().astype(np.float64)
    return (counts + 1.0) / (n_users + 2.0)


def novelty_at_k(recommended: np.ndarray, popularity: np.ndarray) -> float:
    """Mean self-information in bits of the recommended items."""
    if np.any(popularity <= 0) or np.any(popularity > 1):
        raise ValueError("popularity must be probabilities in (0, 1]")
    return float(np.mean(-np.log2(popularity[recommended]))) if recommended.size else 0.0


def ranking_metrics(
    recommended: np.ndarray, relevant: sp.csr_matrix, popularity: np.ndarray, n_items: int
) -> dict[str, float]:
    """All metrics for one list size, keyed by ``metric@k``."""
    k = recommended.shape[1]
    return {
        f"recall@{k}": recall_at_k(recommended, relevant),
        f"ndcg@{k}": ndcg_at_k(recommended, relevant),
        f"precision@{k}": precision_at_k(recommended, relevant),
        f"hit_rate@{k}": hit_rate_at_k(recommended, relevant),
        f"mrr@{k}": mrr_at_k(recommended, relevant),
        f"coverage@{k}": coverage_at_k(recommended, n_items),
        f"novelty@{k}": novelty_at_k(recommended, popularity),
    }
