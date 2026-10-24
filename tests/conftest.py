"""Shared fixtures: small synthetic datasets, splits and fitted models reused across the suite."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from curate.data.interactions import Interactions
from curate.data.split import Split, split_per_user_temporal
from curate.data.synthetic import RetailConfig, RetailWorld, generate_retail
from curate.features.build import build_fit_context
from curate.features.store import FeatureStore
from curate.models.base import FitContext

SMALL = RetailConfig(n_users=150, n_items=60, n_categories=5, latent_dim=8, mean_events_per_user=16.0, seed=7)


@pytest.fixture(autouse=True)
def _limit_threads() -> None:
    torch.set_num_threads(2)


@pytest.fixture(scope="session")
def small_world() -> tuple:
    return generate_retail(SMALL)


@pytest.fixture(scope="session")
def small_dataset(small_world: tuple):
    return small_world[0]


@pytest.fixture(scope="session")
def world(small_world: tuple) -> RetailWorld:
    return small_world[1]


@pytest.fixture(scope="session")
def small_split(small_dataset) -> Split:
    return split_per_user_temporal(small_dataset.interactions)


@pytest.fixture(scope="session")
def small_context(small_dataset, small_split: Split) -> FitContext:
    return build_fit_context(small_dataset, small_split.train, FeatureStore())


@pytest.fixture
def tiny_interactions() -> Interactions:
    users = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    items = np.array([0, 1, 2, 1, 3, 0, 2, 3, 4])
    timestamps = np.array([1, 2, 3, 1, 2, 1, 2, 3, 4])
    return Interactions(users, items, timestamps, np.ones(9, dtype=np.float32), n_users=3, n_items=5)
