"""Off-policy (counterfactual) evaluation of a target policy from logs collected by another policy.

Given logged tuples ``(context, action, propensity, reward)`` where the propensity is the logging
policy's probability of the shown action, the estimators here value a new target policy without
deploying it:

* IPS (inverse propensity scoring): ``mean(w_i r_i)`` with ``w_i = pi(a_i | x_i) / pi_0(a_i | x_i)``.
  Unbiased when the logging policy has full support, but high variance.
* SNIPS (self-normalised IPS): ``sum(w_i r_i) / sum(w_i)``. Biased for finite samples but much lower
  variance; the standard error uses the influence function of the ratio (Swaminathan and Joachims 2015).
* DM (direct method): plug in a reward model ``q(x, a)`` and average it under the target policy.
* DR (doubly robust): DM plus an IPS correction on the residuals ``r_i - q(x_i, a_i)`` (Dudik et al. 2011).

Contexts are user indices and repeat heavily in a log, so every estimator asks the policy and the reward
model for the distinct contexts only and indexes rows back; a 50,000 round log over 943 users costs one
943 by n_items matrix per policy instead of fifty.

Optional weight clipping bounds the variance at the price of bias. Every estimate reports a normal-theory
95 percent interval and the effective sample size ``(sum w)^2 / sum w^2`` so a reader can see when a
policy is too far from the logs to be evaluated from them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

Z_95 = 1.959963984540054


class Policy(Protocol):
    """A stochastic policy over ``n_actions`` actions conditioned on an integer context (user index)."""

    def action_probabilities(self, contexts: np.ndarray) -> np.ndarray:
        """Row-stochastic ``(len(contexts), n_actions)`` matrix."""
        ...


class RewardModel(Protocol):
    """Predicts expected reward for every action of the given contexts."""

    def predict_matrix(self, contexts: np.ndarray) -> np.ndarray:
        """``(len(contexts), n_actions)`` expected rewards."""
        ...


@dataclass(frozen=True)
class BanditFeedback:
    """Logged bandit data; one row per impression."""

    contexts: np.ndarray
    actions: np.ndarray
    propensities: np.ndarray
    rewards: np.ndarray
    n_actions: int

    def __post_init__(self) -> None:
        """Validate shapes and ranges so estimators can assume clean inputs."""
        n = len(self.contexts)
        if not (len(self.actions) == len(self.propensities) == len(self.rewards) == n):
            raise ValueError("all log columns must have the same length")
        if n == 0:
            raise ValueError("empty log")
        if np.any(self.propensities <= 0) or np.any(self.propensities > 1):
            raise ValueError("propensities must lie in (0, 1]")
        if not np.all(np.isfinite(self.rewards)):
            raise ValueError("rewards must be finite")
        if self.actions.min() < 0 or self.actions.max() >= self.n_actions:
            raise ValueError("action index out of range")

    def __len__(self) -> int:
        return len(self.contexts)

    def distinct_contexts(self) -> tuple[np.ndarray, np.ndarray]:
        """Unique contexts and the inverse index mapping every row to its unique context."""
        return np.unique(self.contexts, return_inverse=True)


@dataclass(frozen=True)
class Estimate:
    """A point estimate with its normal-theory interval and diagnostics."""

    estimator: str
    value: float
    std_error: float
    effective_sample_size: float
    n: int

    @property
    def ci_low(self) -> float:
        """Lower end of the 95 percent interval."""
        return self.value - Z_95 * self.std_error

    @property
    def ci_high(self) -> float:
        """Upper end of the 95 percent interval."""
        return self.value + Z_95 * self.std_error

    def as_dict(self) -> dict[str, float | int | str]:
        """Flat mapping for JSON output."""
        return {
            "estimator": self.estimator,
            "value": round(self.value, 6),
            "std_error": round(self.std_error, 6),
            "ci95_low": round(self.ci_low, 6),
            "ci95_high": round(self.ci_high, 6),
            "effective_sample_size": round(self.effective_sample_size, 1),
            "n": self.n,
        }


def _std_error(terms: np.ndarray) -> float:
    n = len(terms)
    return float(terms.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0


def _ess(weights: np.ndarray) -> float:
    total = float(weights.sum())
    return total * total / float((weights**2).sum()) if total > 0 else 0.0


def _logged_action_probabilities(
    feedback: BanditFeedback, target: Policy
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Target probabilities of the logged actions plus the unique-context matrix and inverse index."""
    unique, inverse = feedback.distinct_contexts()
    probs = target.action_probabilities(unique)
    return probs[inverse, feedback.actions], probs, inverse


def importance_weights(feedback: BanditFeedback, target: Policy, clip: float | None = None) -> np.ndarray:
    """``pi(a_i | x_i) / pi_0(a_i | x_i)`` for every logged row, optionally clipped from above."""
    if clip is not None and clip <= 0:
        raise ValueError("clip must be positive")
    logged, _, _ = _logged_action_probabilities(feedback, target)
    weights = logged / feedback.propensities
    return np.minimum(weights, clip) if clip is not None else weights


def ips(feedback: BanditFeedback, target: Policy, clip: float | None = None) -> Estimate:
    """Inverse propensity scoring estimate of the target policy's expected reward."""
    weights = importance_weights(feedback, target, clip)
    terms = weights * feedback.rewards
    return Estimate("ips", float(terms.mean()), _std_error(terms), _ess(weights), len(feedback))


def snips(feedback: BanditFeedback, target: Policy, clip: float | None = None) -> Estimate:
    """Self-normalised IPS; the standard error comes from the ratio's influence function."""
    weights = importance_weights(feedback, target, clip)
    total = float(weights.sum())
    n = len(feedback)
    if total <= 0:
        return Estimate("snips", 0.0, 0.0, 0.0, n)
    value = float((weights * feedback.rewards).sum() / total)
    influence = weights * (feedback.rewards - value) / (total / n)
    return Estimate("snips", value, _std_error(influence), _ess(weights), n)


def direct_method(feedback: BanditFeedback, target: Policy, reward_model: RewardModel) -> Estimate:
    """Average the reward model under the target policy; no importance weights, so no variance blow-up."""
    unique, inverse = feedback.distinct_contexts()
    per_context = (target.action_probabilities(unique) * reward_model.predict_matrix(unique)).sum(axis=1)
    per_row = per_context[inverse]
    return Estimate("dm", float(per_row.mean()), _std_error(per_row), float(len(feedback)), len(feedback))


def doubly_robust(
    feedback: BanditFeedback, target: Policy, reward_model: RewardModel, clip: float | None = None
) -> Estimate:
    """Direct method corrected by importance-weighted residuals of the logged actions."""
    logged, probs, inverse = _logged_action_probabilities(feedback, target)
    unique = np.unique(feedback.contexts)
    predicted = reward_model.predict_matrix(unique)
    weights = logged / feedback.propensities
    if clip is not None:
        weights = np.minimum(weights, clip)
    baseline = (probs * predicted).sum(axis=1)[inverse]
    residual = feedback.rewards - predicted[inverse, feedback.actions]
    terms = baseline + weights * residual
    return Estimate("dr", float(terms.mean()), _std_error(terms), _ess(weights), len(feedback))


def true_policy_value(target: Policy, expected_reward: np.ndarray, contexts: np.ndarray) -> float:
    """Exact value when the expected reward of every (context, action) is known, as in simulation."""
    unique, inverse = np.unique(contexts, return_inverse=True)
    per_context = (target.action_probabilities(unique) * expected_reward[unique]).sum(axis=1)
    return float(per_context[inverse].mean())


def simulate_feedback(
    rng: np.random.Generator, logging_policy: Policy, expected_reward: np.ndarray, contexts: np.ndarray
) -> BanditFeedback:
    """Roll out the logging policy on the given contexts with Bernoulli rewards from ``expected_reward``."""
    unique, inverse = np.unique(contexts, return_inverse=True)
    probs = logging_policy.action_probabilities(unique)
    cumulative = np.cumsum(probs, axis=1)
    cumulative[:, -1] = 1.0
    draws = rng.random(len(contexts))
    actions = np.empty(len(contexts), dtype=np.int64)
    for row in range(len(unique)):
        mask = inverse == row
        actions[mask] = np.searchsorted(cumulative[row], draws[mask], side="right")
    actions = np.minimum(actions, probs.shape[1] - 1)
    rewards = (rng.random(len(contexts)) < expected_reward[contexts, actions]).astype(np.float64)
    return BanditFeedback(contexts, actions, probs[inverse, actions], rewards, probs.shape[1])
