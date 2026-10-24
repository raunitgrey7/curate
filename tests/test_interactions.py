import numpy as np
import pytest

from curate.data.interactions import Catalog, Interactions, UserTable


def test_lengths_must_match():
    with pytest.raises(ValueError, match="same length"):
        Interactions(np.array([0, 1]), np.array([0]), np.array([1, 2]), np.ones(2), 2, 2)


def test_index_bounds_are_checked():
    with pytest.raises(ValueError, match="item index"):
        Interactions(np.array([0]), np.array([5]), np.array([1]), np.ones(1), 1, 2)
    with pytest.raises(ValueError, match="user index"):
        Interactions(np.array([-1]), np.array([0]), np.array([1]), np.ones(1), 1, 2)


def test_non_finite_weights_rejected():
    with pytest.raises(ValueError, match="finite"):
        Interactions(np.array([0]), np.array([0]), np.array([1]), np.array([np.nan]), 1, 1)


def test_to_csr_binary_and_weighted(tiny_interactions):
    binary = tiny_interactions.to_csr()
    assert binary.shape == (3, 5)
    assert binary.nnz == 9
    assert binary.max() == 1.0
    weighted = Interactions(
        np.array([0, 0]), np.array([1, 1]), np.array([1, 2]), np.array([2.0, 3.0], dtype=np.float32), 1, 2
    )
    assert weighted.to_csr(binary=False)[0, 1] == 5.0
    assert weighted.to_csr(binary=True)[0, 1] == 1.0


def test_histories_are_time_ordered_and_distinct():
    inter = Interactions(
        np.array([0, 0, 0, 1]),
        np.array([2, 1, 2, 0]),
        np.array([30, 10, 20, 5]),
        np.ones(4, dtype=np.float32),
        2,
        3,
    )
    histories = inter.histories()
    assert histories[0].tolist() == [1, 2]
    assert histories[1].tolist() == [0]


def test_sorted_by_time_is_stable():
    inter = Interactions(
        np.array([1, 0, 2]), np.array([0, 0, 0]), np.array([5, 5, 1]), np.ones(3, dtype=np.float32), 3, 1
    )
    ordered = inter.sorted_by_time()
    assert ordered.users.tolist() == [2, 1, 0]


def test_counts(tiny_interactions):
    assert tiny_interactions.item_counts().tolist() == [2, 2, 2, 2, 1]
    assert tiny_interactions.user_counts().tolist() == [3, 2, 4]


def test_catalog_alignment_checked():
    with pytest.raises(ValueError, match="aligned"):
        Catalog(np.array([1, 2]), ["a"], np.zeros((2, 1)), ["g"], np.array([0, 0]))


def test_user_table_alignment_checked():
    with pytest.raises(ValueError, match="aligned"):
        UserTable(np.array([1, 2]), np.zeros((2, 2)), ["only_one"])


def test_catalog_genre_labels():
    catalog = Catalog(np.array([1]), ["x"], np.array([[1.0, 0.0, 1.0]]), ["a", "b", "c"], np.array([1999]))
    assert catalog.genre_labels(0) == ["a", "c"]
