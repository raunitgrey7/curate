import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from curate.data.interactions import Interactions
from curate.data.split import split_global_temporal, split_per_user_temporal


def _interactions(users, items, ts, n_items):
    return Interactions(
        np.array(users),
        np.array(items),
        np.array(ts),
        np.ones(len(users), dtype=np.float32),
        max(users) + 1,
        n_items,
    )


def test_per_user_split_holds_out_latest_events():
    inter = _interactions([0] * 10, list(range(10)), list(range(10)), 10)
    split = split_per_user_temporal(inter, valid_frac=0.1, test_frac=0.2)
    assert sorted(split.test.items.tolist()) == [8, 9]
    assert split.valid.items.tolist() == [7]
    assert sorted(split.train.items.tolist()) == list(range(7))
    assert split.test.timestamps.min() > split.valid.timestamps.max() > split.train.timestamps.max()


def test_per_user_split_keeps_min_train():
    inter = _interactions([0, 0, 1], [0, 1, 2], [1, 2, 1], 3)
    split = split_per_user_temporal(inter, valid_frac=0.1, test_frac=0.5, min_train=1)
    assert len(split.train) == 2
    assert len(split.test) == 1
    assert split.test.users.tolist() == [0]


def test_per_user_split_rejects_bad_fractions():
    inter = _interactions([0, 0], [0, 1], [1, 2], 2)
    with pytest.raises(ValueError, match="fractions"):
        split_per_user_temporal(inter, valid_frac=0.5, test_frac=0.6)


def test_global_split_has_no_time_leakage(small_dataset):
    split = split_global_temporal(small_dataset.interactions, valid_frac=0.1, test_frac=0.1)
    assert (
        split.train.timestamps.max()
        < split.valid.timestamps.min()
        <= split.valid.timestamps.max()
        < split.test.timestamps.min()
    )
    assert len(split.train) + len(split.valid) + len(split.test) == len(small_dataset.interactions)


def test_validation_view_swaps_valid_into_test(small_split):
    view = small_split.validation_view()
    assert len(view.test) == len(small_split.valid)
    assert len(view.valid) == 0
    assert view.protocol.endswith(":validation")


def test_seen_before_test_is_binary_union(small_split):
    seen = small_split.seen_before_test()
    assert seen.max() == 1.0
    assert seen.nnz >= small_split.train.to_csr().nnz


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.tuples(st.integers(0, 6), st.integers(0, 9), st.integers(0, 1000)), min_size=1, max_size=120)
)
def test_per_user_split_partitions_rows_and_orders_time(rows):
    users, items, ts = zip(*rows, strict=True)
    inter = Interactions(
        np.array(users), np.array(items), np.array(ts), np.ones(len(rows), dtype=np.float32), 7, 10
    )
    split = split_per_user_temporal(inter)
    assert len(split.train) + len(split.valid) + len(split.test) == len(inter)
    for user in range(7):
        train_t = split.train.timestamps[split.train.users == user]
        test_t = split.test.timestamps[split.test.users == user]
        if len(train_t) and len(test_t):
            assert train_t.max() <= test_t.min()
