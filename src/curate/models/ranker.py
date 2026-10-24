"""Two-stage recommender: retriever candidates re-ranked by a gradient-boosted classifier.

Each retriever proposes its top ``n_candidates`` unseen items; the union is scored by a
``HistGradientBoostingClassifier`` trained to predict whether a candidate appears in the user's next
interactions (the validation window). Features are each retriever's within-user standardised score, rank
and membership flag, the item features from the feature store, the user's activity and the dot product of
the user's genre affinity with the item's genres. The ranker learns how much to trust each retriever for
each kind of user and item, which is where the gain over the best single retriever comes from.

The GBDT runs single-threaded by default: with PyTorch loaded in the same process, scikit-learn's OpenMP
pool contends with PyTorch's and a 1 second fit becomes a 10 second one on macOS (measured in this repo).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from curate.data.interactions import Interactions
from curate.models.base import FitContext, Recommender


@dataclass(frozen=True)
class CandidateTable:
    """Candidates for a batch of users: row ``i`` is the pair ``(users[rows[i]], items[i])``."""

    rows: np.ndarray
    items: np.ndarray
    features: np.ndarray
    feature_names: list[str]


def _dense_mask(seen: sp.csr_matrix, n_items: int) -> np.ndarray:
    mask = np.zeros((seen.shape[0], n_items), dtype=bool)
    coo = seen.tocoo()
    mask[coo.row, coo.col] = True
    return mask


def _group_zscore(values: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    counts = np.bincount(groups, minlength=n_groups).astype(np.float64)
    sums = np.bincount(groups, weights=values, minlength=n_groups)
    means = np.divide(sums, counts, out=np.zeros(n_groups), where=counts > 0)
    squares = np.bincount(groups, weights=(values - means[groups]) ** 2, minlength=n_groups)
    stds = np.sqrt(np.divide(squares, counts, out=np.zeros(n_groups), where=counts > 0))
    return np.divide(values - means[groups], stds[groups], out=np.zeros_like(values), where=stds[groups] > 0)


class TwoStageRecommender:
    """Candidate generation from several retrievers followed by a learned re-ranker."""

    name = "two_stage"

    def __init__(
        self,
        retrievers: dict[str, Recommender],
        *,
        n_candidates: int = 50,
        max_iter: int = 200,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        seed: int = 42,
        n_threads: int = 1,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        self.retrievers = retrievers
        self.n_candidates = n_candidates
        self.max_iter, self.learning_rate, self.max_leaf_nodes, self.seed = (
            max_iter,
            learning_rate,
            max_leaf_nodes,
            seed,
        )
        self.n_threads = n_threads
        self.ranker: HistGradientBoostingClassifier | None = None
        self.feature_names: list[str] = []
        self.context: FitContext | None = None
        self.seen: sp.csr_matrix | None = None
        self.n_training_rows = 0

    def fit(self, train: Interactions, context: FitContext) -> Self:
        """Fit every retriever on ``train`` and remember what each user has already seen."""
        for retriever in self.retrievers.values():
            retriever.fit(train, context)
        self.context = context
        self.seen = train.to_csr()
        return self

    def fit_ranker(self, valid: Interactions, context: FitContext | None = None) -> Self:
        """Train the re-ranker to predict which candidates appear in ``valid``; retrievers must be fit already."""
        if self.seen is None:
            raise RuntimeError("fit must be called before fit_ranker")
        if context is not None:
            self.context = context
        labels_matrix = valid.to_csr()
        users = np.flatnonzero(np.asarray(labels_matrix.sum(axis=1)).ravel() > 0)
        table = self.candidates(users)
        labels = np.asarray(labels_matrix[users[table.rows], table.items]).ravel() > 0
        self.ranker = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.seed,
            early_stopping=False,
        )
        with threadpool_limits(limits=self.n_threads, user_api="openmp"):
            self.ranker.fit(table.features, labels.astype(np.int64))
        self.feature_names = table.feature_names
        self.n_training_rows = len(labels)
        return self

    def with_retrievers(
        self, retrievers: dict[str, Recommender], seen: sp.csr_matrix, context: FitContext
    ) -> Self:
        """Swap in retrievers refit on more data while keeping the trained ranker."""
        if set(retrievers) != set(self.retrievers):
            raise ValueError("retriever names must match the ones the ranker was trained with")
        clone = TwoStageRecommender(
            retrievers,
            n_candidates=self.n_candidates,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            seed=self.seed,
            n_threads=self.n_threads,
        )
        clone.ranker, clone.feature_names, clone.n_training_rows = (
            self.ranker,
            self.feature_names,
            self.n_training_rows,
        )
        clone.context, clone.seen = context, seen
        return clone

    def candidates(self, users: np.ndarray) -> CandidateTable:
        """Union of retriever top-N unseen items with the ranking features for each candidate."""
        if self.seen is None or self.context is None:
            raise RuntimeError("fit must be called before generating candidates")
        n_items = self.seen.shape[1]
        seen = _dense_mask(self.seen[users], n_items)
        scores: dict[str, np.ndarray] = {}
        ranks: dict[str, np.ndarray] = {}
        candidate_mask = np.zeros_like(seen)
        for name, retriever in self.retrievers.items():
            s = retriever.score(users).astype(np.float64)
            s[seen] = -np.inf
            r = np.argsort(np.argsort(-s, axis=1, kind="stable"), axis=1, kind="stable")
            candidate_mask |= r < self.n_candidates
            scores[name], ranks[name] = s, r
        rows, items = np.nonzero(candidate_mask)
        columns, names = self._retriever_columns(scores, ranks, rows, items, len(users))
        side_columns, side_names = self._side_columns(users[rows], items)
        return CandidateTable(rows, items, np.column_stack(columns + side_columns), names + side_names)

    def _retriever_columns(
        self,
        scores: dict[str, np.ndarray],
        ranks: dict[str, np.ndarray],
        rows: np.ndarray,
        items: np.ndarray,
        n_users: int,
    ) -> tuple[list[np.ndarray], list[str]]:
        columns: list[np.ndarray] = []
        names: list[str] = []
        n_sources = np.zeros(len(rows))
        for name in self.retrievers:
            raw = scores[name][rows, items]
            finite = np.where(np.isfinite(raw), raw, np.nan)
            z = _group_zscore(np.nan_to_num(finite, nan=0.0), rows, n_users)
            rank = np.minimum(ranks[name][rows, items], scores[name].shape[1]).astype(np.float64)
            in_top = (rank < self.n_candidates).astype(np.float64)
            n_sources += in_top
            columns += [z, rank, in_top]
            names += [f"{name}_score_z", f"{name}_rank", f"{name}_in_top"]
        columns.append(n_sources)
        names.append("n_sources")
        return columns, names

    def _side_columns(self, users: np.ndarray, items: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
        assert self.context is not None
        ctx = self.context
        item_feats = ctx.item_features[items].astype(np.float64)
        genre_cols = [i for i, n in enumerate(ctx.item_feature_names) if n.startswith("genre_")]
        affinity_cols = [i for i, n in enumerate(ctx.user_feature_names) if n.startswith("affinity_")]
        activity_col = (
            ctx.user_feature_names.index("log_activity_z")
            if "log_activity_z" in ctx.user_feature_names
            else None
        )
        columns = [item_feats[:, i] for i in range(item_feats.shape[1])]
        names = [f"item_{n}" for n in ctx.item_feature_names]
        if genre_cols and len(genre_cols) == len(affinity_cols):
            affinity = (
                ctx.user_features[users][:, affinity_cols] * ctx.item_features[items][:, genre_cols]
            ).sum(axis=1)
            columns.append(affinity.astype(np.float64))
            names.append("user_item_genre_affinity")
        if activity_col is not None:
            columns.append(ctx.user_features[users, activity_col].astype(np.float64))
            names.append("user_log_activity_z")
        return columns, names

    def score(self, users: np.ndarray) -> np.ndarray:
        """Ranker probabilities for candidates and ``-inf`` for everything else."""
        if self.ranker is None or self.seen is None:
            raise RuntimeError("fit_ranker must be called before score")
        table = self.candidates(users)
        with threadpool_limits(limits=self.n_threads, user_api="openmp"):
            probabilities = self.ranker.predict_proba(table.features)[:, 1]
        out = np.full((len(users), self.seen.shape[1]), -np.inf)
        out[table.rows, table.items] = probabilities
        return out

    def feature_importance(self) -> dict[str, float]:
        """Permutation-free proxy: how often each feature is used in splits, normalised to sum to one."""
        if self.ranker is None:
            raise RuntimeError("fit_ranker must be called first")
        counts = np.zeros(len(self.feature_names))
        for stage in self.ranker._predictors:
            for tree in stage:
                nodes = tree.nodes
                internal = nodes[nodes["is_leaf"] == 0]
                np.add.at(counts, internal["feature_idx"], 1)
        total = counts.sum()
        return {
            n: float(c / total) if total else 0.0 for n, c in zip(self.feature_names, counts, strict=True)
        }
