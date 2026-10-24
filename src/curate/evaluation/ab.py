"""A/B experiment simulator with always-valid sequential testing.

Users arrive one at a time, are assigned to control or treatment with equal probability and either
convert or not with a probability that depends on the arm's slate for that user. The simulator runs many
replications at once (vectorised over replications) and applies three analyses at every checkpoint:

* mSPRT, the mixture sequential probability ratio test of Johari, Pekelis and Walsh (2017), which gives an
  always-valid p-value, so stopping as soon as it crosses ``alpha`` keeps the false positive rate at or
  below ``alpha`` however often the experimenter peeks;
* a naive z-test applied at every checkpoint, the common practice the mSPRT exists to replace;
* a fixed-horizon z-test applied once at the end, the textbook baseline.

Measured false positive rates under A/A, power and sample size under a real lift are what the benchmark
reports; nothing here depends on the recommender beyond a per-user conversion probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.stats import norm

from curate.models.base import Recommender, topk_from_scores


def slate_hit_probability(
    model: Recommender, users: np.ndarray, seen: sp.csr_matrix, k: int, expected_reward: np.ndarray
) -> np.ndarray:
    """Probability that at least one item of the user's top-k unseen slate converts."""
    slates = topk_from_scores(model.score(users), k, exclude=seen[users])
    miss = 1.0 - expected_reward[users[:, None], slates]
    return 1.0 - np.prod(miss, axis=1)


@dataclass(frozen=True)
class MixtureSPRT:
    """Two-sample mSPRT for a difference in Bernoulli means with a normal mixing distribution.

    ``tau`` is the standard deviation of the mixing prior on the lift; power is highest when it is close
    to the lift one expects to detect.
    """

    alpha: float = 0.05
    tau: float = 0.02

    def __post_init__(self) -> None:
        """Reject impossible settings."""
        if not 0 < self.alpha < 1 or self.tau <= 0:
            raise ValueError("alpha must be in (0, 1) and tau positive")

    def likelihood_ratio(
        self, n_a: np.ndarray, sum_a: np.ndarray, n_b: np.ndarray, sum_b: np.ndarray
    ) -> np.ndarray:
        """Mixture likelihood ratio of ``H1: lift != 0`` against ``H0: lift = 0``."""
        n_a, n_b = np.maximum(n_a, 1), np.maximum(n_b, 1)
        theta = sum_b / n_b - sum_a / n_a
        pooled = (sum_a + sum_b) / (n_a + n_b)
        variance = np.maximum(pooled * (1.0 - pooled), 1e-12) * (1.0 / n_a + 1.0 / n_b)
        tau2 = self.tau**2
        return np.sqrt(variance / (variance + tau2)) * np.exp(
            tau2 * theta**2 / (2.0 * variance * (variance + tau2))
        )

    def always_valid_p_values(
        self, n_a: np.ndarray, sum_a: np.ndarray, n_b: np.ndarray, sum_b: np.ndarray
    ) -> np.ndarray:
        """Running minimum of ``1 / likelihood ratio`` along the last axis, capped at one."""
        raw = np.minimum(1.0, 1.0 / self.likelihood_ratio(n_a, sum_a, n_b, sum_b))
        return np.minimum.accumulate(raw, axis=-1)


def z_test_p_values(n_a: np.ndarray, sum_a: np.ndarray, n_b: np.ndarray, sum_b: np.ndarray) -> np.ndarray:
    """Two-sided pooled z-test p-values for a difference in proportions."""
    n_a, n_b = np.maximum(n_a, 1), np.maximum(n_b, 1)
    pooled = (sum_a + sum_b) / (n_a + n_b)
    se = np.sqrt(np.maximum(pooled * (1.0 - pooled), 1e-12) * (1.0 / n_a + 1.0 / n_b))
    z = (sum_b / n_b - sum_a / n_a) / se
    return 2.0 * norm.sf(np.abs(z))


@dataclass(frozen=True)
class ExperimentSummary:
    """Aggregate outcome of many replications of one experiment."""

    replications: int
    n_max: int
    true_lift: float
    msprt_reject_rate: float
    msprt_treatment_rate: float
    msprt_mean_n_at_decision: float
    naive_peeking_reject_rate: float
    fixed_horizon_reject_rate: float

    def as_dict(self) -> dict[str, float | int]:
        """Flat mapping for JSON output."""
        return {
            "replications": self.replications,
            "n_max": self.n_max,
            "true_lift": round(self.true_lift, 6),
            "msprt_reject_rate": round(self.msprt_reject_rate, 4),
            "msprt_treatment_better_rate": round(self.msprt_treatment_rate, 4),
            "msprt_mean_n_at_decision": round(self.msprt_mean_n_at_decision, 1),
            "naive_peeking_reject_rate": round(self.naive_peeking_reject_rate, 4),
            "fixed_horizon_reject_rate": round(self.fixed_horizon_reject_rate, 4),
        }


def _checkpoint_sums(
    rng: np.random.Generator,
    control_p: np.ndarray,
    treatment_p: np.ndarray,
    n_max: int,
    replications: int,
    checkpoints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Arm sizes and conversion sums at each checkpoint for every replication."""
    users = rng.integers(0, len(control_p), size=(replications, n_max))
    treated = rng.random((replications, n_max)) < 0.5
    probability = np.where(treated, treatment_p[users], control_p[users])
    converted = rng.random((replications, n_max)) < probability
    n_b = np.cumsum(treated, axis=1)[:, checkpoints - 1]
    n_a = checkpoints[None, :] - n_b
    sum_b = np.cumsum(converted & treated, axis=1)[:, checkpoints - 1]
    sum_a = np.cumsum(converted & ~treated, axis=1)[:, checkpoints - 1]
    return n_a.astype(np.float64), sum_a.astype(np.float64), n_b.astype(np.float64), sum_b.astype(np.float64)


def simulate_experiment(
    rng: np.random.Generator,
    control_p: np.ndarray,
    treatment_p: np.ndarray,
    *,
    n_max: int = 20_000,
    check_every: int = 100,
    replications: int = 400,
    test: MixtureSPRT | None = None,
) -> ExperimentSummary:
    """Run ``replications`` experiments of ``n_max`` arrivals each and summarise the three analyses."""
    if len(control_p) != len(treatment_p) or len(control_p) == 0:
        raise ValueError("control_p and treatment_p must be non-empty and aligned")
    if n_max < check_every or check_every <= 0:
        raise ValueError("n_max must be at least check_every, which must be positive")
    test = test or MixtureSPRT()
    checkpoints = np.arange(check_every, n_max + 1, check_every)
    n_a, sum_a, n_b, sum_b = _checkpoint_sums(rng, control_p, treatment_p, n_max, replications, checkpoints)
    p_always = test.always_valid_p_values(n_a, sum_a, n_b, sum_b)
    crossed = p_always <= test.alpha
    decided = crossed.any(axis=1)
    first = np.where(decided, crossed.argmax(axis=1), len(checkpoints) - 1)
    rows = np.arange(replications)
    lift_at_decision = sum_b[rows, first] / np.maximum(n_b[rows, first], 1) - sum_a[rows, first] / np.maximum(
        n_a[rows, first], 1
    )
    treatment_better = decided & (lift_at_decision > 0)
    naive = (z_test_p_values(n_a, sum_a, n_b, sum_b) <= test.alpha).any(axis=1)
    fixed = z_test_p_values(n_a[:, -1], sum_a[:, -1], n_b[:, -1], sum_b[:, -1]) <= test.alpha
    return ExperimentSummary(
        replications=replications,
        n_max=n_max,
        true_lift=float(treatment_p.mean() - control_p.mean()),
        msprt_reject_rate=float(decided.mean()),
        msprt_treatment_rate=float(treatment_better.mean()),
        msprt_mean_n_at_decision=float(checkpoints[first].mean()),
        naive_peeking_reject_rate=float(naive.mean()),
        fixed_horizon_reject_rate=float(fixed.mean()),
    )
