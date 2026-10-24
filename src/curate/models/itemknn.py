"""Item-based k-nearest-neighbours with shrunk cosine similarity.

Similarity between two items is their co-occurrence count divided by ``sqrt(n_i * n_j) + shrinkage``;
shrinkage pulls similarities computed from a handful of users towards zero (Koren 2010). Only the
``n_neighbors`` strongest neighbours per item are kept. The co-occurrence counts are retained because the
explanation layer uses them to state how many viewers of the anchor item also watched the recommendation.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import scipy.sparse as sp

from curate.data.interactions import Interactions
from curate.models.base import FitContext


class ItemKNNRecommender:
    """Score items by summed similarity to the user's history."""

    name = "itemknn"

    def __init__(self, n_neighbors: int = 100, shrinkage: float = 10.0) -> None:
        if n_neighbors <= 0 or shrinkage < 0:
            raise ValueError("n_neighbors must be positive and shrinkage non-negative")
        self.n_neighbors = n_neighbors
        self.shrinkage = shrinkage
        self.similarity: sp.csr_matrix | None = None
        self.cooccurrence: sp.csr_matrix | None = None
        self.item_counts: np.ndarray | None = None
        self._train: sp.csr_matrix | None = None

    def fit(self, train: Interactions, context: FitContext) -> Self:  # noqa: ARG002
        """Build the shrunk cosine similarity and keep the top neighbours per item."""
        matrix = train.to_csr()
        self._train = matrix
        cooccurrence = (matrix.T @ matrix).tocsr()
        cooccurrence.setdiag(0)
        cooccurrence.eliminate_zeros()
        counts = np.asarray(matrix.sum(axis=0)).ravel()
        self.item_counts = counts
        self.cooccurrence = cooccurrence
        coo = cooccurrence.tocoo()
        denominator = np.sqrt(counts[coo.row] * counts[coo.col]) + self.shrinkage
        sims = sp.csr_matrix((coo.data / denominator, (coo.row, coo.col)), shape=cooccurrence.shape)
        self.similarity = _keep_top_neighbors(sims, self.n_neighbors)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        """Sum of similarities between each candidate and the user's training items."""
        if self.similarity is None or self._train is None:
            raise RuntimeError("fit must be called before score")
        return np.asarray((self._train[users] @ self.similarity).todense(), dtype=np.float64)

    def similar_items(self, item: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        """The ``k`` most similar items to ``item`` with their similarities, strongest first."""
        if self.similarity is None:
            raise RuntimeError("fit must be called before similar_items")
        row = self.similarity.getrow(item)
        order = np.argsort(-row.data, kind="stable")[:k]
        return row.indices[order], row.data[order]


def _keep_top_neighbors(sims: sp.csr_matrix, n_neighbors: int) -> sp.csr_matrix:
    """Keep only the ``n_neighbors`` largest entries in each row."""
    sims = sims.tocsr()
    rows, cols, vals = [], [], []
    for i in range(sims.shape[0]):
        start, end = sims.indptr[i], sims.indptr[i + 1]
        data = sims.data[start:end]
        if len(data) > n_neighbors:
            keep = np.argpartition(-data, n_neighbors - 1)[:n_neighbors]
        else:
            keep = np.arange(len(data))
        rows.append(np.full(len(keep), i))
        cols.append(sims.indices[start:end][keep])
        vals.append(data[keep])
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=sims.shape,
        dtype=np.float64,
    )
