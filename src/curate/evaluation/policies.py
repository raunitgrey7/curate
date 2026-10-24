"""Policies built from recommenders or popularity, with full support so logs can be re-weighted.

A policy maps a user to a distribution over items. Items the user has already seen get probability zero in
both logging and target policies, which keeps the comparison about new items. An ``epsilon`` share of mass
is spread uniformly over unseen items so every target policy is supported by the logging policy.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from curate.models.base import Recommender


def _unseen_mask(seen: sp.csr_matrix | None, users: np.ndarray, n_items: int) -> np.ndarray:
    allowed = np.ones((len(users), n_items), dtype=bool)
    if seen is not None:
        coo = seen[users].tocoo()
        allowed[coo.row, coo.col] = False
    return allowed


def _mix_uniform(probs: np.ndarray, allowed: np.ndarray, epsilon: float) -> np.ndarray:
    uniform = allowed / np.maximum(allowed.sum(axis=1, keepdims=True), 1)
    mixed = (1.0 - epsilon) * probs + epsilon * uniform
    return mixed / mixed.sum(axis=1, keepdims=True)


class SoftmaxPolicy:
    """Boltzmann distribution over a recommender's scores for unseen items, mixed with uniform exploration."""

    def __init__(
        self,
        model: Recommender,
        seen: sp.csr_matrix | None,
        n_items: int,
        temperature: float = 1.0,
        epsilon: float = 0.05,
    ) -> None:
        if temperature <= 0 or not 0 <= epsilon <= 1:
            raise ValueError("temperature must be positive and epsilon in [0, 1]")
        self.model, self.seen, self.n_items, self.temperature, self.epsilon = (
            model,
            seen,
            n_items,
            temperature,
            epsilon,
        )

    def action_probabilities(self, contexts: np.ndarray) -> np.ndarray:
        """Softmax over standardised scores, restricted to unseen items."""
        scores = np.asarray(self.model.score(contexts), dtype=np.float64)
        allowed = _unseen_mask(self.seen, contexts, self.n_items)
        scores = np.where(allowed & np.isfinite(scores), scores, np.nan)
        centre = np.nanmean(scores, axis=1, keepdims=True)
        spread = np.nanstd(scores, axis=1, keepdims=True)
        z = (scores - centre) / np.where(spread > 0, spread, 1.0)
        logits = np.where(np.isnan(z), -np.inf, z / self.temperature)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return _mix_uniform(probs, allowed, self.epsilon)


class TopKPolicy:
    """Uniform over a recommender's top-k unseen items (one slot shown at random), mixed with exploration."""

    def __init__(
        self, model: Recommender, seen: sp.csr_matrix | None, n_items: int, k: int = 10, epsilon: float = 0.05
    ) -> None:
        if k <= 0 or not 0 <= epsilon <= 1:
            raise ValueError("k must be positive and epsilon in [0, 1]")
        self.model, self.seen, self.n_items, self.k, self.epsilon = model, seen, n_items, k, epsilon

    def action_probabilities(self, contexts: np.ndarray) -> np.ndarray:
        """Probability ``1/k`` on each top-k unseen item before exploration mixing."""
        scores = np.asarray(self.model.score(contexts), dtype=np.float64)
        allowed = _unseen_mask(self.seen, contexts, self.n_items)
        scores = np.where(allowed, scores, -np.inf)
        order = np.argsort(-scores, axis=1, kind="stable")[:, : self.k]
        probs = np.zeros_like(scores)
        np.put_along_axis(probs, order, 1.0 / self.k, axis=1)
        probs *= allowed
        probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
        return _mix_uniform(probs, allowed, self.epsilon)


class UniformPolicy:
    """Uniform over unseen items; the weakest reasonable logging policy and a useful control."""

    def __init__(self, seen: sp.csr_matrix | None, n_items: int) -> None:
        self.seen, self.n_items = seen, n_items

    def action_probabilities(self, contexts: np.ndarray) -> np.ndarray:
        """Equal mass on every unseen item."""
        allowed = _unseen_mask(self.seen, contexts, self.n_items)
        return allowed / allowed.sum(axis=1, keepdims=True)


class TabularRewardModel:
    """Reward model backed by a dense ``(n_contexts, n_actions)`` table of predicted rewards."""

    def __init__(self, table: np.ndarray) -> None:
        self.table = np.asarray(table, dtype=np.float64)

    def predict_matrix(self, contexts: np.ndarray) -> np.ndarray:
        """Rows of the table for the given contexts."""
        return self.table[contexts]
