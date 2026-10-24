"""Feature builders: turn a dataset plus training interactions into stored user and item features.

Item features: genre flags, a decade one-hot and standardised log popularity. User features: the profile
columns from the dataset, standardised log activity, and the user's genre affinity (the normalised genre
distribution of their training items). Everything is computed from training data only so the features can
be rebuilt for any split without leaking the future.
"""

from __future__ import annotations

import numpy as np

from curate.data.interactions import Dataset, Interactions
from curate.features.store import FeatureStore
from curate.models.base import FitContext

DECADES = (1950, 1960, 1970, 1980, 1990, 2000)
USER_SET = "user_v1"
ITEM_SET = "item_v1"


def _standardise(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    return (values - values.mean()) / std if std > 0 else np.zeros_like(values)


def item_feature_matrix(dataset: Dataset, train: Interactions) -> tuple[list[str], np.ndarray]:
    """Genres, decade one-hot and standardised log popularity for every item."""
    catalog = dataset.catalog
    decade_names = (
        [f"decade_lt_{DECADES[0]}"] + [f"decade_{d}s" for d in DECADES[:-1]] + [f"decade_ge_{DECADES[-1]}"]
    )
    decades = np.zeros((catalog.n_items, len(decade_names)), dtype=np.float32)
    known = catalog.years > 0
    decades[
        np.arange(catalog.n_items)[known], np.searchsorted(DECADES, catalog.years[known], side="right")
    ] = 1.0
    popularity = _standardise(np.log1p(train.item_counts().astype(np.float32)))[:, None]
    names = [f"genre_{g}" for g in catalog.genre_names] + decade_names + ["log_popularity_z"]
    return names, np.hstack([catalog.genres.astype(np.float32), decades, popularity.astype(np.float32)])


def user_feature_matrix(dataset: Dataset, train: Interactions) -> tuple[list[str], np.ndarray]:
    """Profile columns, standardised log activity and genre affinity for every user."""
    csr = train.to_csr()
    genre_counts = np.asarray(csr @ dataset.catalog.genres, dtype=np.float32)
    totals = genre_counts.sum(axis=1, keepdims=True)
    affinity = np.divide(genre_counts, totals, out=np.zeros_like(genre_counts), where=totals > 0)
    activity = _standardise(np.log1p(np.asarray(csr.sum(axis=1)).ravel().astype(np.float32)))[:, None]
    names = (
        list(dataset.users.feature_names)
        + ["log_activity_z"]
        + [f"affinity_{g}" for g in dataset.catalog.genre_names]
    )
    return names, np.hstack(
        [dataset.users.features.astype(np.float32), activity.astype(np.float32), affinity]
    )


def write_catalog_metadata(dataset: Dataset, store: FeatureStore) -> None:
    """Store titles, genres, years and external ids so serving can describe items and users."""
    catalog = dataset.catalog
    store.write_metadata(
        "item",
        np.arange(catalog.n_items),
        [
            {
                "item_id": int(catalog.item_ids[i]),
                "title": catalog.titles[i],
                "genres": catalog.genre_labels(i),
                "year": int(catalog.years[i]),
            }
            for i in range(catalog.n_items)
        ],
    )
    store.write_metadata(
        "user", np.arange(dataset.users.n_users), [{"user_id": int(u)} for u in dataset.users.user_ids]
    )


def build_fit_context(dataset: Dataset, train: Interactions, store: FeatureStore) -> FitContext:
    """Compute features from training data, persist them in the store and read them back as a ``FitContext``."""
    item_names, item_matrix = item_feature_matrix(dataset, train)
    user_names, user_matrix = user_feature_matrix(dataset, train)
    store.write_features("item", ITEM_SET, np.arange(dataset.catalog.n_items), item_names, item_matrix)
    store.write_features("user", USER_SET, np.arange(dataset.users.n_users), user_names, user_matrix)
    write_catalog_metadata(dataset, store)
    _, item_names, item_matrix = store.read_features("item", ITEM_SET)
    _, user_names, user_matrix = store.read_features("user", USER_SET)
    return FitContext(dataset.catalog, dataset.users, item_matrix, item_names, user_matrix, user_names)
