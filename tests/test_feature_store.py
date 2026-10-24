import numpy as np
import pytest

from curate.features.build import (
    ITEM_SET,
    USER_SET,
    build_fit_context,
    item_feature_matrix,
    user_feature_matrix,
)
from curate.features.store import FeatureStore


def test_round_trip_and_versions(tmp_path):
    store = FeatureStore(tmp_path / "f.sqlite")
    ids = np.array([3, 1, 2])
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    store.write_features("item", "v1", ids, ["a", "b"], matrix)
    out_ids, names, out = store.read_features("item", "v1")
    assert out_ids.tolist() == [1, 2, 3]
    assert names == ["a", "b"]
    assert out.tolist() == [[3.0, 4.0], [5.0, 6.0], [1.0, 2.0]]
    assert store.get_vector("item", "v1", 3) == {"a": 1.0, "b": 2.0}
    assert store.get_vector("item", "v1", 99) is None
    store.write_features("item", "v1", ids[:1], ["a", "b"], matrix[:1])
    assert store.read_features("item", "v1")[2].shape == (1, 2)
    assert [fs.name for fs in store.list_feature_sets()] == ["v1"]
    store.close()


def test_persisted_store_reopens(tmp_path):
    path = tmp_path / "f.sqlite"
    store = FeatureStore(path)
    store.write_features("user", "v1", np.array([0]), ["x"], np.array([[7.0]], dtype=np.float32))
    store.write_metadata("user", np.array([0]), [{"user_id": 42}])
    store.close()
    reopened = FeatureStore(path)
    assert reopened.get_vector("user", "v1", 0) == {"x": 7.0}
    assert reopened.get_metadata("user", 0) == {"user_id": 42}
    assert reopened.count("user") == 1
    reopened.close()


def test_write_validation():
    store = FeatureStore()
    with pytest.raises(ValueError, match="shape"):
        store.write_features("item", "v1", np.array([0, 1]), ["a"], np.zeros((1, 1)))
    with pytest.raises(ValueError, match="non-finite"):
        store.write_features("item", "v1", np.array([0]), ["a"], np.array([[np.nan]]))
    with pytest.raises(KeyError):
        store.read_features("item", "missing")
    with pytest.raises(ValueError, match="same length"):
        store.write_metadata("item", np.array([0, 1]), [{}])


def test_metadata_round_trips_unicode():
    store = FeatureStore()
    store.write_metadata("item", np.array([5]), [{"title": "Caf\xe9 à la crème", "genres": ["Drama"]}])
    assert store.get_metadata("item", 5)["title"] == "Caf\xe9 à la crème"
    assert store.get_metadata("item", 6) is None


def test_feature_builders_use_training_data_only(small_dataset, small_split):
    names, matrix = item_feature_matrix(small_dataset, small_split.train)
    assert matrix.shape == (small_dataset.catalog.n_items, len(names))
    assert names[-1] == "log_popularity_z"
    assert abs(float(matrix[:, -1].mean())) < 1e-5
    unames, umatrix = user_feature_matrix(small_dataset, small_split.train)
    assert umatrix.shape == (small_dataset.users.n_users, len(unames))
    affinity = umatrix[:, [i for i, n in enumerate(unames) if n.startswith("affinity_")]]
    assert np.allclose(affinity.sum(axis=1), 1.0)


def test_build_fit_context_reads_back_from_store(small_dataset, small_split):
    store = FeatureStore()
    ctx = build_fit_context(small_dataset, small_split.train, store)
    assert {fs.name for fs in store.list_feature_sets()} == {ITEM_SET, USER_SET}
    assert ctx.item_features.shape[0] == small_dataset.catalog.n_items
    assert store.get_metadata("item", 0)["title"] == small_dataset.catalog.titles[0]
    assert store.count("user") == small_dataset.users.n_users
