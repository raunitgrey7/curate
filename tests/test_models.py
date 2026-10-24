import numpy as np
import pytest
import scipy.sparse as sp

from curate.evaluation.offline import EvaluationTarget, evaluate_model
from curate.models import (
    BPRRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    TwoTowerRecommender,
    topk_from_scores,
)
from curate.models.mf import sample_negatives, unique_pairs
from curate.models.ranker import TwoStageRecommender

BPR_KW = {"dim": 16, "epochs": 8, "n_threads": 2}
TT_KW = {"dim": 16, "id_dim": 8, "hidden": 32, "epochs": 6, "n_threads": 2}


def _recall(model, split):
    target = EvaluationTarget.from_split(split)
    return evaluate_model(model, target, ks=(10,)).metrics["recall@10"]


def test_topk_masks_seen_and_breaks_ties_by_index():
    scores = np.array([[1.0, 1.0, 1.0, 0.5]])
    exclude = sp.csr_matrix(np.array([[1, 0, 0, 0]]))
    assert topk_from_scores(scores, 2, exclude).tolist() == [[1, 2]]
    assert topk_from_scores(scores, 3).tolist() == [[0, 1, 2]]
    with pytest.raises(ValueError, match="k must be"):
        topk_from_scores(scores, 0)
    with pytest.raises(ValueError, match="k must be"):
        topk_from_scores(scores, 5)


def test_popularity_orders_by_count(tiny_interactions, small_context):
    model = PopularityRecommender().fit(tiny_interactions, small_context)
    scores = model.score(np.array([0, 1]))
    assert scores.shape == (2, 5)
    assert np.array_equal(scores[0], scores[1])
    assert scores[0, 4] < scores[0, 0]
    decayed = PopularityRecommender(half_life_days=1.0).fit(tiny_interactions, small_context)
    assert decayed.score(np.array([0])).shape == (1, 5)


def test_unfitted_models_raise():
    for model in (
        PopularityRecommender(),
        ItemKNNRecommender(),
        BPRRecommender(**BPR_KW),
        TwoTowerRecommender(**TT_KW),
    ):
        with pytest.raises(RuntimeError, match="fit"):
            model.score(np.array([0]))


def test_itemknn_similarity_properties(small_split, small_context):
    model = ItemKNNRecommender(n_neighbors=5, shrinkage=0.0).fit(small_split.train, small_context)
    sims = model.similarity
    assert sims.shape == (small_split.train.n_items,) * 2
    assert sims.diagonal().sum() == 0.0
    assert sims.max() <= 1.0 + 1e-9
    assert max(np.diff(sims.indptr)) <= 5
    neighbours, values = model.similar_items(0, 3)
    assert len(neighbours) <= 3
    assert np.all(np.diff(values) <= 0)
    with pytest.raises(ValueError, match="n_neighbors"):
        ItemKNNRecommender(n_neighbors=0)


def test_itemknn_scores_only_from_history(tiny_interactions, small_context):
    model = ItemKNNRecommender(n_neighbors=10, shrinkage=0.0).fit(tiny_interactions, small_context)
    scores = model.score(np.array([1]))
    assert scores.shape == (1, 5)
    assert scores[0, 4] > 0  # user 1 watched items 1 and 3, item 4 co-occurs with 3 through user 2


def test_unique_pairs_and_negative_sampling(tiny_interactions):
    users, items = unique_pairs(tiny_interactions)
    assert len(users) == 9
    keys = np.sort(users * 5 + items)
    rng = np.random.default_rng(0)
    negatives = sample_negatives(rng, users, keys, 5, max_rounds=100)
    assert not np.isin(users * 5 + negatives, keys).any()


def test_bpr_learns_and_beats_random(small_split, small_context):
    model = BPRRecommender(**BPR_KW).fit(small_split.train, small_context)
    assert model.loss_history[0] > model.loss_history[-1]
    assert model.user_factors.shape == (small_split.train.n_users, 16)
    recall = _recall(model, small_split)
    random_recall = _recall(_Random(small_split.train.n_items), small_split)
    assert recall > random_recall


def test_two_tower_learns_and_beats_random(small_split, small_context):
    model = TwoTowerRecommender(**TT_KW).fit(small_split.train, small_context)
    assert model.loss_history[0] > model.loss_history[-1]
    norms = np.linalg.norm(model.item_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)
    assert _recall(model, small_split) > _recall(_Random(small_split.train.n_items), small_split)


def test_two_tower_without_logq_still_fits(small_split, small_context):
    model = TwoTowerRecommender(logq_correction=False, **TT_KW).fit(small_split.train, small_context)
    assert model.score(np.array([0])).shape == (1, small_split.train.n_items)


class _Random:
    name = "random"

    def __init__(self, n_items: int) -> None:
        self.n_items = n_items

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.random.default_rng(0).random((len(users), self.n_items))


def test_bpr_is_deterministic_for_the_same_seed(small_split, small_context):
    a = BPRRecommender(seed=5, **BPR_KW).fit(small_split.train, small_context)
    b = BPRRecommender(seed=5, **BPR_KW).fit(small_split.train, small_context)
    assert np.array_equal(a.item_factors, b.item_factors)
    assert a.loss_history == b.loss_history
    c = BPRRecommender(seed=6, **BPR_KW).fit(small_split.train, small_context)
    assert not np.array_equal(a.item_factors, c.item_factors)


def test_two_tower_is_deterministic_for_the_same_seed(small_split, small_context):
    a = TwoTowerRecommender(seed=5, **TT_KW).fit(small_split.train, small_context)
    b = TwoTowerRecommender(seed=5, **TT_KW).fit(small_split.train, small_context)
    assert np.array_equal(a.item_embeddings, b.item_embeddings)
    target = EvaluationTarget.from_split(small_split)
    from curate.evaluation.offline import recommend_for_users

    assert np.array_equal(recommend_for_users(a, target, 5), recommend_for_users(b, target, 5))


def _two_stage(split, context, seed=0):
    retrievers = {
        "popularity": PopularityRecommender(),
        "itemknn": ItemKNNRecommender(n_neighbors=20),
        "bpr": BPRRecommender(seed=seed, **BPR_KW),
    }
    model = TwoStageRecommender(retrievers, n_candidates=15, max_iter=30, seed=seed, n_threads=2).fit(
        split.train, context
    )
    return model.fit_ranker(split.valid)


def test_two_stage_scores_candidates_only_and_is_deterministic(small_split, small_context):
    model = _two_stage(small_split, small_context)
    users = np.arange(5)
    scores = model.score(users)
    finite = np.isfinite(scores)
    assert finite.sum(axis=1).min() >= 15
    assert np.all((scores[finite] >= 0) & (scores[finite] <= 1))
    assert model.n_training_rows > 0
    again = _two_stage(small_split, small_context)
    assert np.array_equal(again.score(users), scores)
    importance = model.feature_importance()
    assert pytest.approx(sum(importance.values()), abs=1e-6) == 1.0
    assert set(importance) == set(model.feature_names)


def test_two_stage_candidate_features_have_expected_columns(small_split, small_context):
    model = _two_stage(small_split, small_context)
    table = model.candidates(np.arange(3))
    assert table.features.shape[1] == len(table.feature_names)
    assert "itemknn_rank" in table.feature_names
    assert "n_sources" in table.feature_names
    assert "user_item_genre_affinity" in table.feature_names
    assert np.all(np.isfinite(table.features))
    assert table.rows.max() == 2


def test_two_stage_with_retrievers_requires_same_names(small_split, small_context):
    model = _two_stage(small_split, small_context)
    with pytest.raises(ValueError, match="retriever names"):
        model.with_retrievers(
            {"popularity": PopularityRecommender()}, small_split.seen_before_test(), small_context
        )
    with pytest.raises(RuntimeError):
        TwoStageRecommender({"popularity": PopularityRecommender()}).score(np.array([0]))
    with pytest.raises(ValueError, match="at least one"):
        TwoStageRecommender({})


def test_evaluate_model_excludes_seen_items(small_split, small_context):
    model = PopularityRecommender().fit(small_split.train, small_context)
    target = EvaluationTarget.from_split(small_split)
    from curate.evaluation.offline import recommend_for_users

    recommended = recommend_for_users(model, target, 10)
    seen = target.seen[target.users]
    rows = np.repeat(np.arange(len(target.users)), 10)
    assert np.asarray(seen[rows, recommended.ravel()]).sum() == 0
    result = evaluate_model(model, target, ks=(5, 10))
    assert result.n_eval_users == len(target.users)
    assert "recall@5" in result.metrics
    assert "ndcg@10" in result.metrics
    payload = result.as_dict()
    assert payload["model"] == "popularity"
