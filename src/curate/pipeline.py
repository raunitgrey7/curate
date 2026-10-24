"""Training pipeline shared by the benchmark and the serving bundle.

The two-stage model needs labels that follow the retrievers' training window, so the pipeline always
fits the retrievers on the training window, trains the ranker on the validation window, then refits the
retrievers on everything available (train plus validation for evaluation, all interactions for serving)
and reuses the trained ranker on top of them. Every model receives the same history at test time, which
keeps the comparison between single retrievers and the two-stage system fair.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from curate import __version__
from curate.data.interactions import Dataset, Interactions
from curate.data.split import Split, split_per_user_temporal
from curate.features.build import build_fit_context
from curate.features.store import FeatureStore
from curate.models.base import FitContext, Recommender
from curate.models.itemknn import ItemKNNRecommender
from curate.models.mf import BPRRecommender
from curate.models.popularity import PopularityRecommender
from curate.models.ranker import TwoStageRecommender
from curate.models.two_tower import TwoTowerRecommender

RETRIEVER_NAMES = ("popularity", "itemknn", "bpr", "two_tower")
BUNDLE_FILE = "bundle.pkl"
STORE_FILE = "features.sqlite"
MANIFEST_FILE = "manifest.json"


def make_retrievers(seed: int, n_threads: int = 4) -> dict[str, Recommender]:
    """Fresh, unfitted retrievers with the hyperparameters chosen on the validation split."""
    return {
        "popularity": PopularityRecommender(),
        "itemknn": ItemKNNRecommender(n_neighbors=100, shrinkage=10.0),
        "bpr": BPRRecommender(seed=seed, n_threads=n_threads),
        "two_tower": TwoTowerRecommender(seed=seed, n_threads=n_threads),
    }


def concat_interactions(first: Interactions, second: Interactions) -> Interactions:
    """Union of two interaction sets over the same index space."""
    return Interactions(
        users=np.concatenate([first.users, second.users]),
        items=np.concatenate([first.items, second.items]),
        timestamps=np.concatenate([first.timestamps, second.timestamps]),
        weights=np.concatenate([first.weights, second.weights]),
        n_users=first.n_users,
        n_items=first.n_items,
    )


@dataclass
class TrainedModels:
    """Retrievers refit on the final history plus the two-stage model that uses them."""

    retrievers: dict[str, Recommender]
    two_stage: TwoStageRecommender
    context: FitContext
    fit_seconds: dict[str, float] = field(default_factory=dict)

    def all_models(self) -> dict[str, Recommender]:
        """Every model keyed by name, two-stage last."""
        return {**self.retrievers, self.two_stage.name: self.two_stage}


def fit_two_stage(
    dataset: Dataset, split: Split, store: FeatureStore, seed: int, n_threads: int = 4
) -> tuple[TwoStageRecommender, float]:
    """Fit retrievers on train and the ranker on validation; returns the model and the elapsed seconds."""
    start = time.perf_counter()
    context = build_fit_context(dataset, split.train, store)
    two_stage = TwoStageRecommender(make_retrievers(seed, n_threads), seed=seed).fit(split.train, context)
    two_stage.fit_ranker(split.valid)
    return two_stage, time.perf_counter() - start


def refit_on(
    dataset: Dataset,
    history: Interactions,
    store: FeatureStore,
    two_stage: TwoStageRecommender,
    seed: int,
    n_threads: int = 4,
) -> TrainedModels:
    """Refit every retriever on ``history`` and attach them to the already trained ranker."""
    context = build_fit_context(dataset, history, store)
    retrievers = make_retrievers(seed, n_threads)
    timings: dict[str, float] = {}
    for name, model in retrievers.items():
        start = time.perf_counter()
        model.fit(history, context)
        timings[name] = time.perf_counter() - start
    final = two_stage.with_retrievers(retrievers, history.to_csr(), context)
    return TrainedModels(retrievers, final, context, timings)


def fit_for_evaluation(
    dataset: Dataset, split: Split, store: FeatureStore, seed: int, n_threads: int = 4
) -> TrainedModels:
    """Models ready to be scored on ``split.test``: retrievers see train plus validation."""
    two_stage, ranker_seconds = fit_two_stage(dataset, split, store, seed, n_threads)
    models = refit_on(
        dataset, concat_interactions(split.train, split.valid), store, two_stage, seed, n_threads
    )
    models.fit_seconds["two_stage"] = ranker_seconds + sum(models.fit_seconds.values())
    return models


@dataclass
class ServingBundle:
    """Everything the API needs at request time, persisted as one pickle plus the sqlite store."""

    dataset_name: str
    models: dict[str, Recommender]
    history: sp.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray
    seed: int
    created_at: float
    version: str = __version__
    train_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def n_items(self) -> int:
        """Catalogue size."""
        return int(self.history.shape[1])

    def save(self, artifact_dir: Path) -> None:
        """Write the pickle and a human-readable manifest next to the feature store."""
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with (artifact_dir / BUNDLE_FILE).open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        manifest: dict[str, Any] = {
            "dataset": self.dataset_name,
            "models": sorted(self.models),
            "n_users": int(self.history.shape[0]),
            "n_items": self.n_items,
            "seed": self.seed,
            "created_at": self.created_at,
            "version": self.version,
            "train_seconds": {k: round(v, 3) for k, v in self.train_seconds.items()},
        }
        (artifact_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")

    @classmethod
    def load(cls, artifact_dir: Path) -> ServingBundle:
        """Load a bundle written by ``save``; only our own artifacts are ever unpickled."""
        path = artifact_dir / BUNDLE_FILE
        if not path.exists():
            raise FileNotFoundError(f"no serving bundle at {path}; run `curate train` first")
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
        if not isinstance(bundle, cls):
            raise TypeError("artifact is not a ServingBundle")
        return bundle


def fit_for_serving(dataset: Dataset, artifact_dir: Path, seed: int, n_threads: int = 4) -> ServingBundle:
    """Train the full bundle on every interaction and persist it with its feature store."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    store_path = artifact_dir / STORE_FILE
    store_path.unlink(missing_ok=True)
    store = FeatureStore(store_path)
    try:
        split = split_per_user_temporal(dataset.interactions)
        two_stage, ranker_seconds = fit_two_stage(dataset, split, store, seed, n_threads)
        models = refit_on(dataset, dataset.interactions, store, two_stage, seed, n_threads)
    finally:
        store.close()
    bundle = ServingBundle(
        dataset_name=dataset.name,
        models=models.all_models(),
        history=dataset.interactions.to_csr(),
        user_ids=dataset.users.user_ids,
        item_ids=dataset.catalog.item_ids,
        seed=seed,
        created_at=time.time(),
        train_seconds={**models.fit_seconds, "ranker": ranker_seconds},
    )
    bundle.save(artifact_dir)
    return bundle
