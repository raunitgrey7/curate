import numpy as np
import pytest

from curate.data.split import split_per_user_temporal
from curate.features.store import FeatureStore
from curate.pipeline import (
    ServingBundle,
    concat_interactions,
    fit_for_evaluation,
    fit_for_serving,
    make_retrievers,
)


def test_make_retrievers_names_and_seed():
    retrievers = make_retrievers(seed=3)
    assert set(retrievers) == {"popularity", "itemknn", "bpr", "two_tower"}
    assert retrievers["bpr"].seed == 3


def test_concat_interactions(small_split):
    merged = concat_interactions(small_split.train, small_split.valid)
    assert len(merged) == len(small_split.train) + len(small_split.valid)
    assert merged.n_items == small_split.train.n_items


@pytest.mark.slow
def test_fit_for_evaluation_and_serving_round_trip(small_dataset, tmp_path, monkeypatch):
    from curate import pipeline

    def fast_retrievers(seed, n_threads=2):
        from curate.models import (
            BPRRecommender,
            ItemKNNRecommender,
            PopularityRecommender,
            TwoTowerRecommender,
        )

        return {
            "popularity": PopularityRecommender(),
            "itemknn": ItemKNNRecommender(n_neighbors=20),
            "bpr": BPRRecommender(dim=16, epochs=5, seed=seed, n_threads=n_threads),
            "two_tower": TwoTowerRecommender(
                dim=16, id_dim=8, hidden=32, epochs=4, seed=seed, n_threads=n_threads
            ),
        }

    monkeypatch.setattr(pipeline, "make_retrievers", fast_retrievers)
    split = split_per_user_temporal(small_dataset.interactions)
    models = fit_for_evaluation(small_dataset, split, FeatureStore(), seed=1)
    assert set(models.all_models()) == {"popularity", "itemknn", "bpr", "two_tower", "two_stage"}
    assert models.fit_seconds["two_stage"] >= models.fit_seconds["bpr"]
    scores = models.two_stage.score(np.array([0, 1]))
    assert scores.shape == (2, small_dataset.catalog.n_items)

    bundle = fit_for_serving(small_dataset, tmp_path, seed=1)
    for name in ("bundle.pkl", "features.sqlite", "manifest.json"):
        assert (tmp_path / name).exists()
    loaded = ServingBundle.load(tmp_path)
    assert loaded.dataset_name == "retail_synthetic"
    assert sorted(loaded.models) == sorted(bundle.models)
    assert loaded.history.shape == (small_dataset.users.n_users, small_dataset.catalog.n_items)
    a = bundle.models["two_stage"].score(np.array([3]))
    b = loaded.models["two_stage"].score(np.array([3]))
    assert np.array_equal(a, b)


def test_load_missing_bundle_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="curate train"):
        ServingBundle.load(tmp_path)
