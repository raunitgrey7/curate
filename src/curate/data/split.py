"""Time-based splits.

Two protocols are provided. ``split_per_user_temporal`` holds out the last fraction of each user's own
timeline, which keeps every user evaluable; it is the primary protocol because 68 percent of MovieLens
100k users rated everything within a single day, so a single global cut leaves about one hundred users
with history on both sides. ``split_global_temporal`` is the stricter protocol with no cross-user
leakage and is reported alongside as a secondary result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from curate.data.interactions import Interactions


@dataclass(frozen=True)
class Split:
    """Train, validation and test interactions produced by one protocol."""

    train: Interactions
    valid: Interactions
    test: Interactions
    protocol: str

    def seen_before_test(self) -> sp.csr_matrix:
        """Binary user by item matrix of everything observed before the test window."""
        seen = self.train.to_csr() + self.valid.to_csr()
        seen.data[:] = 1.0
        return seen

    def validation_view(self) -> Split:
        """The same split with validation playing the role of test, for model selection without touching test."""
        empty = self.valid.take(np.zeros(0, dtype=np.int64))
        return Split(self.train, empty, self.valid, f"{self.protocol}:validation")

    def summary(self) -> dict[str, int | str]:
        """Row counts for reporting."""
        return {
            "protocol": self.protocol,
            "n_train": len(self.train),
            "n_valid": len(self.valid),
            "n_test": len(self.test),
        }


def _per_user_assignment(n: int, valid_frac: float, test_frac: float, min_train: int) -> tuple[int, int]:
    """Numbers of validation and test rows for a user with ``n`` events."""
    n_test = max(1, round(n * test_frac)) if n > min_train else 0
    n_valid = round(n * valid_frac)
    if n - n_test - n_valid < min_train:
        n_valid = max(0, n - n_test - min_train)
    return n_valid, n_test


def split_per_user_temporal(
    interactions: Interactions, valid_frac: float = 0.1, test_frac: float = 0.2, min_train: int = 1
) -> Split:
    """Hold out the chronologically last ``test_frac`` of each user's events, and ``valid_frac`` before that."""
    if not 0 <= valid_frac < 1 or not 0 < test_frac < 1 or valid_frac + test_frac >= 1:
        raise ValueError("fractions must satisfy 0 <= valid < 1, 0 < test < 1 and valid + test < 1")
    ordered = interactions.sorted_by_time()
    order = np.argsort(ordered.users, kind="stable")
    labels = np.zeros(len(ordered), dtype=np.int8)
    boundaries = np.flatnonzero(np.diff(ordered.users[order])) + 1
    for chunk in np.split(order, boundaries):
        if not len(chunk):
            continue
        n_valid, n_test = _per_user_assignment(len(chunk), valid_frac, test_frac, min_train)
        if n_test:
            labels[chunk[-n_test:]] = 2
        if n_valid:
            labels[chunk[len(chunk) - n_test - n_valid : len(chunk) - n_test]] = 1
    return Split(
        ordered.take(labels == 0), ordered.take(labels == 1), ordered.take(labels == 2), "per_user_temporal"
    )


def split_global_temporal(
    interactions: Interactions, valid_frac: float = 0.1, test_frac: float = 0.1
) -> Split:
    """Cut the whole log at two timestamp quantiles; no event in train happens after an event in test."""
    if not 0 <= valid_frac < 1 or not 0 < test_frac < 1 or valid_frac + test_frac >= 1:
        raise ValueError("fractions must satisfy 0 <= valid < 1, 0 < test < 1 and valid + test < 1")
    ordered = interactions.sorted_by_time()
    n = len(ordered)
    test_start = int(np.floor(n * (1.0 - test_frac)))
    valid_start = int(np.floor(n * (1.0 - test_frac - valid_frac)))
    t_test = ordered.timestamps[test_start]
    t_valid = ordered.timestamps[valid_start]
    ts = ordered.timestamps
    return Split(
        ordered.take(ts < t_valid),
        ordered.take((ts >= t_valid) & (ts < t_test)),
        ordered.take(ts >= t_test),
        "global_temporal",
    )
