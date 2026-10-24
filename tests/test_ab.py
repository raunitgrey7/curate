import numpy as np
import pytest

from curate.evaluation.ab import MixtureSPRT, simulate_experiment, slate_hit_probability, z_test_p_values
from curate.models.popularity import PopularityRecommender


def test_msprt_validation():
    with pytest.raises(ValueError, match="alpha"):
        MixtureSPRT(alpha=1.5)
    with pytest.raises(ValueError, match="alpha"):
        MixtureSPRT(tau=0.0)


def test_likelihood_ratio_is_one_at_no_evidence_limit():
    test = MixtureSPRT(tau=0.02)
    n = np.array([10.0])
    ratio = test.likelihood_ratio(n, np.array([5.0]), n, np.array([5.0]))
    assert ratio[0] < 1.0
    assert test.always_valid_p_values(n, np.array([5.0]), n, np.array([5.0]))[0] == 1.0


def test_always_valid_p_values_are_running_minimum():
    test = MixtureSPRT()
    n = np.array([[100.0, 200.0, 300.0]])
    sum_a = np.array([[20.0, 40.0, 60.0]])
    sum_b = np.array([[30.0, 50.0, 60.0]])
    p = test.always_valid_p_values(n, sum_a, n, sum_b)
    assert np.all(np.diff(p[0]) <= 1e-12)


def test_z_test_matches_textbook_value():
    p = z_test_p_values(np.array([1000.0]), np.array([100.0]), np.array([1000.0]), np.array([130.0]))
    assert p[0] == pytest.approx(0.0355, abs=0.0005)


def test_aa_false_positive_rate_controlled_and_naive_inflated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.3, size=500)
    summary = simulate_experiment(rng, p, p, n_max=5000, check_every=50, replications=300)
    assert summary.true_lift == 0.0
    assert summary.msprt_reject_rate <= 0.05 + 0.03
    assert summary.naive_peeking_reject_rate > 0.15
    assert summary.fixed_horizon_reject_rate <= 0.05 + 0.04


def test_large_lift_is_detected_early():
    rng = np.random.default_rng(1)
    control = rng.uniform(0.05, 0.2, size=500)
    treatment = np.clip(control * 1.5, 0, 1)
    summary = simulate_experiment(rng, control, treatment, n_max=10_000, check_every=100, replications=100)
    assert summary.msprt_treatment_rate > 0.9
    assert summary.msprt_mean_n_at_decision < 10_000


def test_simulate_experiment_validation():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="aligned"):
        simulate_experiment(rng, np.array([0.1]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="check_every"):
        simulate_experiment(rng, np.array([0.1]), np.array([0.1]), n_max=10, check_every=100)


def test_slate_hit_probability(small_split, small_context, world):
    model = PopularityRecommender().fit(small_split.train, small_context)
    users = np.arange(10)
    probs = slate_hit_probability(model, users, small_split.seen_before_test(), 5, world.click_matrix())
    assert probs.shape == (10,)
    assert np.all((probs >= 0) & (probs <= 1))
    one = slate_hit_probability(model, users, small_split.seen_before_test(), 1, world.click_matrix())
    assert np.all(probs >= one - 1e-12)
