"""Core containers: interactions in a dense index space plus item and user side tables.

External ids (MovieLens movie numbers, SKUs) are mapped to contiguous indices once at load time so every
model can use arrays and sparse matrices directly. The containers are frozen dataclasses; derived views
(histories, sparse matrices) are computed on demand rather than cached, which keeps them trivially
picklable and cheap to validate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class Interactions:
    """Feedback events ``(user, item, timestamp, weight)`` in internal index space.

    ``weights`` carry the rating for explicit data or an event weight for implicit data; the models in
    this package treat every row as a positive signal and use the weight only where documented.
    """

    users: np.ndarray
    items: np.ndarray
    timestamps: np.ndarray
    weights: np.ndarray
    n_users: int
    n_items: int

    def __post_init__(self) -> None:
        """Validate shapes, dtypes and index bounds so later code can trust the arrays."""
        n = len(self.users)
        if not (len(self.items) == len(self.timestamps) == len(self.weights) == n):
            raise ValueError("users, items, timestamps and weights must have the same length")
        if n and (self.users.min() < 0 or self.users.max() >= self.n_users):
            raise ValueError("user index out of range")
        if n and (self.items.min() < 0 or self.items.max() >= self.n_items):
            raise ValueError("item index out of range")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("weights must be finite")

    def __len__(self) -> int:
        return len(self.users)

    def sorted_by_time(self) -> Interactions:
        """Return a copy ordered by timestamp with a stable tie-break on the original position."""
        order = np.argsort(self.timestamps, kind="stable")
        return self.take(order)

    def take(self, index: np.ndarray) -> Interactions:
        """Select rows by position or boolean mask."""
        return Interactions(
            users=self.users[index],
            items=self.items[index],
            timestamps=self.timestamps[index],
            weights=self.weights[index],
            n_users=self.n_users,
            n_items=self.n_items,
        )

    def to_csr(self, binary: bool = True) -> sp.csr_matrix:
        """User by item sparse matrix; duplicates are summed unless ``binary`` clips them to one."""
        values = np.ones(len(self), dtype=np.float32) if binary else self.weights.astype(np.float32)
        matrix = sp.csr_matrix((values, (self.users, self.items)), shape=(self.n_users, self.n_items))
        matrix.sum_duplicates()
        if binary:
            matrix.data[:] = 1.0
        return matrix

    def histories(self) -> list[np.ndarray]:
        """Per-user arrays of distinct item indices in time order (first occurrence kept)."""
        ordered = self.sorted_by_time()
        histories: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(self.n_users)]
        split_points = np.flatnonzero(np.diff(np.sort(ordered.users, kind="stable"))) + 1
        order = np.argsort(ordered.users, kind="stable")
        for chunk in np.split(order, split_points):
            if len(chunk):
                items = ordered.items[chunk]
                _, first = np.unique(items, return_index=True)
                histories[int(ordered.users[chunk[0]])] = items[np.sort(first)]
        return histories

    def item_counts(self) -> np.ndarray:
        """Number of distinct users per item."""
        return np.asarray(self.to_csr().sum(axis=0)).ravel()

    def user_counts(self) -> np.ndarray:
        """Number of distinct items per user."""
        return np.asarray(self.to_csr().sum(axis=1)).ravel()


@dataclass(frozen=True)
class Catalog:
    """Item side information aligned with the internal item index."""

    item_ids: np.ndarray
    titles: list[str]
    genres: np.ndarray
    genre_names: list[str]
    years: np.ndarray

    def __post_init__(self) -> None:
        """Check that every per-item array has one row per item."""
        n = len(self.item_ids)
        if len(self.titles) != n or self.genres.shape != (n, len(self.genre_names)) or len(self.years) != n:
            raise ValueError("catalog arrays are not aligned")

    @property
    def n_items(self) -> int:
        """Number of items in the catalogue."""
        return len(self.item_ids)

    def genre_labels(self, item: int) -> list[str]:
        """Names of the genres flagged for one item."""
        return [name for name, flag in zip(self.genre_names, self.genres[item], strict=True) if flag]


@dataclass(frozen=True)
class UserTable:
    """User side information aligned with the internal user index."""

    user_ids: np.ndarray
    features: np.ndarray
    feature_names: list[str]

    def __post_init__(self) -> None:
        """Check feature matrix shape against ids and names."""
        if self.features.shape != (len(self.user_ids), len(self.feature_names)):
            raise ValueError("user feature matrix is not aligned with ids and names")

    @property
    def n_users(self) -> int:
        """Number of users."""
        return len(self.user_ids)


@dataclass(frozen=True)
class Dataset:
    """A named bundle of interactions and side tables plus provenance metadata."""

    name: str
    interactions: Interactions
    catalog: Catalog
    users: UserTable
    provenance: dict[str, str]

    def __post_init__(self) -> None:
        """Check that the index spaces agree."""
        if (
            self.interactions.n_items != self.catalog.n_items
            or self.interactions.n_users != self.users.n_users
        ):
            raise ValueError("interactions, catalog and users disagree on index sizes")

    def user_index(self) -> dict[int, int]:
        """Map external user id to internal index."""
        return {int(ext): idx for idx, ext in enumerate(self.users.user_ids)}

    def item_index(self) -> dict[int, int]:
        """Map external item id to internal index."""
        return {int(ext): idx for idx, ext in enumerate(self.catalog.item_ids)}
