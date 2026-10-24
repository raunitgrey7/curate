import numpy as np
import pytest

from curate.data.synthetic import RetailConfig, generate_retail


def test_generator_is_deterministic():
    a, _ = generate_retail(RetailConfig(n_users=80, n_items=40, seed=3))
    b, _ = generate_retail(RetailConfig(n_users=80, n_items=40, seed=3))
    assert np.array_equal(a.interactions.users, b.interactions.users)
    assert np.array_equal(a.interactions.items, b.interactions.items)
    assert np.array_equal(a.interactions.timestamps, b.interactions.timestamps)


def test_different_seeds_differ():
    a, _ = generate_retail(RetailConfig(n_users=80, n_items=40, seed=3))
    b, _ = generate_retail(RetailConfig(n_users=80, n_items=40, seed=4))
    assert not np.array_equal(a.interactions.items, b.interactions.items)


def test_shapes_and_alignment(small_dataset, world):
    inter = small_dataset.interactions
    assert inter.n_users == small_dataset.users.n_users == world.utility.shape[0]
    assert inter.n_items == small_dataset.catalog.n_items == world.utility.shape[1]
    assert small_dataset.catalog.genres.sum(axis=1).tolist() == [1.0] * inter.n_items


def test_every_user_has_minimum_events(small_dataset):
    counts = np.bincount(small_dataset.interactions.users, minlength=small_dataset.interactions.n_users)
    assert counts.min() >= 5


def test_popularity_is_heavy_tailed():
    dataset, _ = generate_retail(RetailConfig(seed=11))
    counts = np.sort(dataset.interactions.item_counts())[::-1]
    top_decile = counts[: len(counts) // 10].sum() / counts.sum()
    assert top_decile > 0.2
    assert counts[0] > 3 * np.median(counts)


def test_click_rate_is_calibrated(world):
    assert abs(world.click_matrix().mean() - 0.05) < 0.005


def test_items_appear_after_release(small_dataset, world):
    inter = small_dataset.interactions
    day = (inter.timestamps - 1_704_067_200) / 86_400
    assert np.all(day >= world.item_release_day[inter.items] - 1e-6)


def test_funnel_weights_are_graded(small_dataset):
    values, counts = np.unique(small_dataset.interactions.weights, return_counts=True)
    assert values.tolist() == [1.0, 2.0, 3.0]
    assert counts[0] > counts[2]


def test_invalid_config_rejected():
    with pytest.raises(ValueError, match="positive"):
        RetailConfig(n_users=0)
    with pytest.raises(ValueError, match="min_events"):
        RetailConfig(n_items=10, min_events_per_user=10)
    with pytest.raises(ValueError, match="target_click_rate"):
        RetailConfig(target_click_rate=1.5)
