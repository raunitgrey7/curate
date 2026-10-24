import numpy as np
import pytest

from curate.config import Settings
from curate.explain.engine import Explainer
from curate.explain.llm import llm_available, reword_explanation
from curate.features.build import build_fit_context
from curate.features.store import FeatureStore
from curate.models import ItemKNNRecommender, TwoTowerRecommender


@pytest.fixture(scope="module")
def explainer(small_dataset, small_split):
    store = FeatureStore()
    context = build_fit_context(small_dataset, small_split.train, store)
    itemknn = ItemKNNRecommender(n_neighbors=20).fit(small_split.train, context)
    two_tower = TwoTowerRecommender(dim=16, id_dim=8, hidden=32, epochs=3, n_threads=2).fit(
        small_split.train, context
    )
    return Explainer(itemknn, two_tower, small_split.train.to_csr(), store, "retail_synthetic")


def test_because_you_bought_uses_measured_support(explainer, small_split, small_dataset):
    history = small_split.train.to_csr()[0].indices
    assert explainer.itemknn.similarity is not None
    candidates = np.asarray(explainer.itemknn.similarity[history].sum(axis=0)).ravel()
    candidates[history] = -1
    item = int(np.argmax(candidates))
    explanation = explainer.explain(0, item)
    assert explanation.kind == "because_you_watched"
    assert explanation.anchor_item_id is not None
    assert explanation.text.startswith("Because you bought")
    assert 0.0 <= explanation.support <= 1.0
    assert f"{explanation.support:.0%}" in explanation.text
    anchor_index = int(np.flatnonzero(small_dataset.catalog.item_ids == explanation.anchor_item_id)[0])
    expected = (
        explainer.itemknn.cooccurrence[anchor_index, item] / explainer.itemknn.item_counts[anchor_index]
    )
    assert explanation.support == pytest.approx(expected)


def test_cold_user_gets_popularity_explanation(explainer):
    explanation = explainer.explain(None, 3)
    assert explanation.kind == "popular"
    assert "people bought this" in explanation.text
    assert explanation.as_dict()["shared_genres"] == []


def test_embedding_fallback_when_knn_has_no_link(explainer, small_split):
    import scipy.sparse as sp

    history = sp.csr_matrix(([1.0], ([0], [0])), shape=small_split.train.to_csr().shape)
    isolated = Explainer(explainer.itemknn, explainer.two_tower, history, explainer.store, "retail_synthetic")
    sims = np.asarray(explainer.itemknn.similarity[0].todense()).ravel()
    unlinked = int(np.flatnonzero((sims <= 0) & (np.arange(len(sims)) != 0))[0])
    explanation = isolated.explain(0, unlinked)
    assert explanation.kind == "similar_taste"
    assert "similar taste profile" in explanation.text


def test_verb_depends_on_dataset(explainer, small_split):
    movie = Explainer(explainer.itemknn, None, small_split.train.to_csr(), explainer.store, "movielens_100k")
    assert movie.verb == "watched"
    assert (
        Explainer(explainer.itemknn, None, small_split.train.to_csr(), explainer.store, "other").verb
        == "chose"
    )


def test_unfitted_itemknn_rejected(small_split, explainer):
    with pytest.raises(ValueError, match="fitted"):
        Explainer(ItemKNNRecommender(), None, small_split.train.to_csr(), explainer.store, "retail_synthetic")


def test_llm_rewording_is_off_by_default_and_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(llm_explanations=True)
    assert not llm_available(settings)
    assert reword_explanation("Because you watched X.", settings) == "Because you watched X."
    assert reword_explanation("text", Settings(llm_explanations=False)) == "text"
