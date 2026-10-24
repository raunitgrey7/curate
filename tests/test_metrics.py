import numpy as np
import pytest
import scipy.sparse as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from curate.evaluation import metrics as m


def _relevant(rows):
    """rows: list of lists of relevant item ids, n_items = 10."""
    matrix = sp.lil_matrix((len(rows), 10))
    for u, items in enumerate(rows):
        for i in items:
            matrix[u, i] = 1
    return matrix.tocsr()


def test_recall_precision_hit_rate_known_values():
    rec = np.array([[0, 1, 2], [5, 6, 7]])
    rel = _relevant([[0, 2, 9, 8], [1]])
    assert m.recall_at_k(rec, rel) == pytest.approx((2 / 4 + 0) / 2)
    assert m.precision_at_k(rec, rel) == pytest.approx((2 / 3 + 0) / 2)
    assert m.hit_rate_at_k(rec, rel) == pytest.approx(0.5)
    assert m.mrr_at_k(rec, rel) == pytest.approx(0.5)


def test_ndcg_perfect_and_partial():
    rel = _relevant([[0, 1], [3]])
    assert m.ndcg_at_k(np.array([[0, 1, 2], [3, 4, 5]]), rel) == pytest.approx(1.0)
    partial = m.ndcg_at_k(np.array([[2, 0, 1], [4, 5, 3]]), rel)
    expected_u0 = (1 / np.log2(3) + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
    expected_u1 = (1 / np.log2(4)) / 1
    assert partial == pytest.approx((expected_u0 + expected_u1) / 2)


def test_users_without_relevant_items_are_skipped_in_recall():
    rel = _relevant([[0], []])
    assert m.recall_at_k(np.array([[0, 1], [2, 3]]), rel) == 1.0
    assert m.ndcg_at_k(np.array([[0, 1], [2, 3]]), rel) == 1.0


def test_coverage_and_novelty():
    rec = np.array([[0, 1], [1, 2]])
    assert m.coverage_at_k(rec, 10) == pytest.approx(0.3)
    popularity = np.full(10, 0.5)
    assert m.novelty_at_k(rec, popularity) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="probabilities"):
        m.novelty_at_k(rec, np.zeros(10))
    with pytest.raises(ValueError, match="positive"):
        m.coverage_at_k(rec, 0)


def test_item_popularity_is_smoothed():
    train = sp.csr_matrix(np.array([[1, 0], [1, 0], [0, 0]]))
    pop = m.item_popularity(train)
    assert pop.tolist() == pytest.approx([(2 + 1) / 5, 1 / 5])
    assert np.all(pop > 0)


def test_hit_matrix_requires_2d():
    with pytest.raises(ValueError, match="2-D"):
        m.hit_matrix(np.array([0, 1]), _relevant([[0]]))


def test_ranking_metrics_keys():
    out = m.ranking_metrics(np.array([[0, 1]]), _relevant([[1]]), np.full(10, 0.1), 10)
    assert set(out) == {"recall@2", "ndcg@2", "precision@2", "hit_rate@2", "mrr@2", "coverage@2", "novelty@2"}


@settings(max_examples=50, deadline=None)
@given(
    st.lists(st.lists(st.integers(0, 9), min_size=0, max_size=5), min_size=1, max_size=6),
    st.integers(1, 10),
    st.integers(0, 10_000),
)
def test_metrics_are_bounded_and_recall_monotone_in_k(rows, k, seed):
    rng = np.random.default_rng(seed)
    rec = np.vstack([rng.permutation(10) for _ in rows])
    rel = _relevant(rows)
    for fn in (m.recall_at_k, m.precision_at_k, m.hit_rate_at_k, m.mrr_at_k, m.ndcg_at_k):
        value = fn(rec[:, :k], rel)
        assert 0.0 <= value <= 1.0 + 1e-12
    if k < 10:
        assert m.recall_at_k(rec[:, : k + 1], rel) >= m.recall_at_k(rec[:, :k], rel) - 1e-12
