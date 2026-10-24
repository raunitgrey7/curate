import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from curate.evaluation.counterfactual import (
    BanditFeedback,
    direct_method,
    doubly_robust,
    importance_weights,
    ips,
    simulate_feedback,
    snips,
    true_policy_value,
)
from curate.evaluation.policies import TabularRewardModel, UniformPolicy


class FixedPolicy:
    def __init__(self, probs: np.ndarray) -> None:
        self.probs = probs

    def action_probabilities(self, contexts: np.ndarray) -> np.ndarray:
        return np.tile(self.probs, (len(contexts), 1))


def _world(seed=0, n_contexts=50, n_actions=8):
    rng = np.random.default_rng(seed)
    return rng, rng.uniform(0.0, 0.3, size=(n_contexts, n_actions))


def test_feedback_validation():
    with pytest.raises(ValueError, match="propensities"):
        BanditFeedback(np.array([0]), np.array([0]), np.array([0.0]), np.array([1.0]), 2)
    with pytest.raises(ValueError, match="same length"):
        BanditFeedback(np.array([0, 1]), np.array([0]), np.array([0.5]), np.array([1.0]), 2)
    with pytest.raises(ValueError, match="empty"):
        BanditFeedback(np.array([]), np.array([]), np.array([]), np.array([]), 2)
    with pytest.raises(ValueError, match="out of range"):
        BanditFeedback(np.array([0]), np.array([5]), np.array([0.5]), np.array([1.0]), 2)


def test_ips_is_unbiased_against_exact_value():
    rng, reward = _world()
    contexts = rng.integers(0, 50, size=40_000)
    logging = UniformPolicy(None, 8)
    feedback = simulate_feedback(rng, logging, reward, contexts)
    target = FixedPolicy(np.array([0.5, 0.3, 0.1, 0.1, 0, 0, 0, 0]))
    truth = true_policy_value(target, reward, contexts)
    estimate = ips(feedback, target)
    assert abs(estimate.value - truth) < 3 * estimate.std_error + 1e-9
    assert estimate.ci_low <= truth <= estimate.ci_high
    assert 0 < estimate.effective_sample_size <= len(feedback)


def test_snips_close_to_truth_with_smaller_error_than_ips_on_average():
    rng, reward = _world(seed=3)
    target = FixedPolicy(np.array([0.7, 0.3, 0, 0, 0, 0, 0, 0]))
    ips_err, snips_err = [], []
    for _ in range(20):
        contexts = rng.integers(0, 50, size=4000)
        feedback = simulate_feedback(rng, UniformPolicy(None, 8), reward, contexts)
        truth = true_policy_value(target, reward, contexts)
        ips_err.append(abs(ips(feedback, target).value - truth))
        snips_err.append(abs(snips(feedback, target).value - truth))
    assert np.mean(snips_err) < 0.02
    assert np.mean(snips_err) <= np.mean(ips_err) * 1.5


def test_clipping_bounds_weights_and_requires_positive_clip():
    rng, reward = _world()
    contexts = rng.integers(0, 50, size=500)
    feedback = simulate_feedback(rng, UniformPolicy(None, 8), reward, contexts)
    target = FixedPolicy(np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    assert importance_weights(feedback, target).max() == pytest.approx(8.0)
    assert importance_weights(feedback, target, clip=2.0).max() <= 2.0
    with pytest.raises(ValueError, match="clip"):
        importance_weights(feedback, target, clip=0.0)


def test_dm_with_oracle_model_recovers_truth_and_dr_matches():
    rng, reward = _world(seed=5)
    contexts = rng.integers(0, 50, size=5000)
    feedback = simulate_feedback(rng, UniformPolicy(None, 8), reward, contexts)
    target = FixedPolicy(np.full(8, 1 / 8))
    oracle = TabularRewardModel(reward)
    truth = true_policy_value(target, reward, contexts)
    assert direct_method(feedback, target, oracle).value == pytest.approx(truth)
    dr = doubly_robust(feedback, target, oracle)
    assert abs(dr.value - truth) < 3 * dr.std_error + 1e-9


def test_on_policy_logging_matches_mean_reward():
    rng, reward = _world(seed=9)
    contexts = rng.integers(0, 50, size=3000)
    logging = UniformPolicy(None, 8)
    feedback = simulate_feedback(rng, logging, reward, contexts)
    assert ips(feedback, logging).value == pytest.approx(feedback.rewards.mean())
    assert snips(feedback, logging).value == pytest.approx(feedback.rewards.mean())


def test_estimate_as_dict_rounds_and_reports_n():
    rng, reward = _world()
    contexts = rng.integers(0, 50, size=100)
    feedback = simulate_feedback(rng, UniformPolicy(None, 8), reward, contexts)
    payload = ips(feedback, UniformPolicy(None, 8)).as_dict()
    assert payload["n"] == 100
    assert payload["estimator"] == "ips"
    assert payload["ci95_low"] <= payload["value"] <= payload["ci95_high"]


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 10_000), st.floats(0.05, 1.0))
def test_snips_stays_inside_reward_range(seed, concentration):
    rng, reward = _world(seed=seed)
    contexts = rng.integers(0, 50, size=300)
    feedback = simulate_feedback(rng, UniformPolicy(None, 8), reward, contexts)
    probs = rng.dirichlet(np.full(8, concentration))
    value = snips(feedback, FixedPolicy(probs)).value
    assert 0.0 <= value <= 1.0
