"""Seeded synthetic implicit-feedback generator for a retail catalogue with known ground truth.

The generator is statistically motivated rather than arbitrary:

* Item exposure follows a Zipf law (``popularity_exponent``), the long-tail shape observed in retail sales
  and media consumption (Anderson 2006; Clauset, Shalizi and Newman 2009).
* User activity is Gamma distributed, so event counts are Gamma-Poisson (negative binomial) mixtures,
  the standard model for over-dispersed purchase counts (Ehrenberg 1959, the NBD model).
* Preferences are low-rank: users and items live in a shared latent space built from category centroids
  plus noise, which is the assumption behind every factorisation model we evaluate, with a popularity
  term added so that the data has the exposure bias real logs have.
* Items are chosen without replacement per user with the Gumbel-top-k trick, which samples exactly
  from the softmax over utilities (Kool, van Hoof and Welling 2019).
* Each view passes through a two-step funnel (cart, then purchase) whose probabilities increase with the
  latent utility, giving graded implicit weights.

Because the latent utility is known, a ``RetailWorld`` can compute exact click probabilities, which is
what makes the counterfactual estimators and the A/B simulator testable against ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from curate.data.interactions import Catalog, Dataset, Interactions, UserTable

SECONDS_PER_DAY = 86_400
BASE_EPOCH = 1_704_067_200  # 2024-01-01T00:00:00Z, a fixed origin for synthetic timestamps


@dataclass(frozen=True)
class RetailConfig:
    """Knobs of the generator; the defaults produce about 80k events in a few seconds."""

    n_users: int = 2000
    n_items: int = 600
    n_categories: int = 12
    n_regions: int = 4
    latent_dim: int = 16
    horizon_days: int = 180
    popularity_exponent: float = 1.0
    popularity_weight: float = 0.6
    mean_events_per_user: float = 40.0
    activity_shape: float = 1.5
    min_events_per_user: int = 5
    preference_scale: float = 2.0
    target_click_rate: float = 0.05
    seed: int = 42

    def __post_init__(self) -> None:
        """Reject configurations that cannot produce a valid dataset."""
        if min(self.n_users, self.n_items, self.n_categories, self.latent_dim, self.horizon_days) <= 0:
            raise ValueError("sizes must be positive")
        if self.min_events_per_user < 1 or self.min_events_per_user >= self.n_items:
            raise ValueError("min_events_per_user must be in [1, n_items)")
        if not 0 < self.target_click_rate < 1:
            raise ValueError("target_click_rate must be in (0, 1)")


@dataclass(frozen=True)
class RetailWorld:
    """Ground truth behind a synthetic dataset: latent utilities and the implied click model."""

    utility: np.ndarray
    click_scale: float
    click_threshold: float
    item_release_day: np.ndarray
    user_segment: np.ndarray
    item_category: np.ndarray

    def click_probability(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Probability that each (user, item) pair produces a positive reward when shown."""
        logits = self.click_scale * (self.utility[users, items] - self.click_threshold)
        return 1.0 / (1.0 + np.exp(-logits))

    def click_matrix(self) -> np.ndarray:
        """Dense (n_users, n_items) matrix of click probabilities."""
        logits = self.click_scale * (self.utility - self.click_threshold)
        return 1.0 / (1.0 + np.exp(-logits))


def _sample_items(rng: np.random.Generator, utility: np.ndarray, counts: np.ndarray) -> list[np.ndarray]:
    """Gumbel-top-k: distinct items per user drawn from the softmax over utilities."""
    chosen: list[np.ndarray] = []
    for user, count in enumerate(counts):
        perturbed = utility[user] + rng.gumbel(size=utility.shape[1])
        top = np.argpartition(-perturbed, count - 1)[:count]
        chosen.append(top[np.argsort(-perturbed[top])])
    return chosen


def _calibrate_threshold(utility: np.ndarray, scale: float, target: float) -> float:
    """Pick the logistic threshold so the mean click probability over all pairs equals ``target``."""
    lo, hi = float(utility.min()) - 10.0, float(utility.max()) + 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        mean_prob = float(np.mean(1.0 / (1.0 + np.exp(-scale * (utility - mid)))))
        lo, hi = (mid, hi) if mean_prob > target else (lo, mid)
    return 0.5 * (lo + hi)


def _latent_space(
    rng: np.random.Generator, cfg: RetailConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centroids = rng.normal(size=(cfg.n_categories, cfg.latent_dim))
    category_shares = rng.dirichlet(np.full(cfg.n_categories, 2.0))
    item_category = rng.choice(cfg.n_categories, size=cfg.n_items, p=category_shares)
    item_vectors = centroids[item_category] + 0.5 * rng.normal(size=(cfg.n_items, cfg.latent_dim))
    user_segment = rng.choice(cfg.n_categories, size=cfg.n_users, p=category_shares)
    user_vectors = 0.8 * centroids[user_segment] + 0.6 * rng.normal(size=(cfg.n_users, cfg.latent_dim))
    return user_vectors, item_vectors, user_segment, item_category


def generate_retail(cfg: RetailConfig | None = None) -> tuple[Dataset, RetailWorld]:
    """Generate a seeded retail dataset and the ground-truth world that produced it."""
    cfg = cfg or RetailConfig()
    rng = np.random.default_rng(cfg.seed)
    user_vectors, item_vectors, user_segment, item_category = _latent_space(rng, cfg)
    ranks = rng.permutation(cfg.n_items) + 1
    log_popularity = -cfg.popularity_exponent * np.log(ranks)
    affinity = (user_vectors @ item_vectors.T) / np.sqrt(cfg.latent_dim)
    utility = (cfg.preference_scale * affinity + cfg.popularity_weight * log_popularity).astype(np.float32)
    release_day = rng.uniform(-365.0, cfg.horizon_days * 0.5, size=cfg.n_items)
    rates = rng.gamma(cfg.activity_shape, cfg.mean_events_per_user / cfg.activity_shape, size=cfg.n_users)
    counts = np.clip(rng.poisson(rates), cfg.min_events_per_user, cfg.n_items // 2)
    start_day = rng.uniform(0.0, cfg.horizon_days * 0.6, size=cfg.n_users)
    chosen = _sample_items(rng, utility, counts)
    users = np.repeat(np.arange(cfg.n_users), counts)
    items = np.concatenate(chosen)
    earliest = np.maximum(start_day[users], release_day[items])
    earliest = np.minimum(earliest, cfg.horizon_days - 1.0)
    event_day = rng.uniform(earliest, cfg.horizon_days)
    timestamps = BASE_EPOCH + (event_day * SECONDS_PER_DAY).astype(np.int64)
    chosen_utility = utility[users, items]
    centred = chosen_utility - chosen_utility.mean()
    cart = rng.random(len(users)) < 1.0 / (1.0 + np.exp(-(centred - 0.5)))
    purchase = cart & (rng.random(len(users)) < 0.4)
    weights = (1.0 + cart + purchase).astype(np.float32)
    interactions = Interactions(users, items, timestamps, weights, cfg.n_users, cfg.n_items).sorted_by_time()
    threshold = _calibrate_threshold(utility, 1.5, cfg.target_click_rate)
    world = RetailWorld(utility, 1.5, threshold, release_day, user_segment, item_category)
    dataset = Dataset(
        name="retail_synthetic",
        interactions=interactions,
        catalog=_catalog(cfg, item_category, release_day),
        users=_users(rng, cfg, user_segment),
        provenance={
            "generator": "curate.data.synthetic.generate_retail",
            **{k: str(v) for k, v in asdict(cfg).items()},
        },
    )
    return dataset, world


def _catalog(cfg: RetailConfig, item_category: np.ndarray, release_day: np.ndarray) -> Catalog:
    genre_names = [f"category_{k:02d}" for k in range(cfg.n_categories)]
    genres = np.zeros((cfg.n_items, cfg.n_categories), dtype=np.float32)
    genres[np.arange(cfg.n_items), item_category] = 1.0
    titles = [f"SKU-{i:04d} ({genre_names[c]})" for i, c in enumerate(item_category)]
    years = 2024 + np.floor(release_day / 365.0).astype(np.int64)
    return Catalog(
        item_ids=np.arange(cfg.n_items) + 1000,
        titles=titles,
        genres=genres,
        genre_names=genre_names,
        years=years,
    )


def _users(rng: np.random.Generator, cfg: RetailConfig, user_segment: np.ndarray) -> UserTable:
    region = rng.integers(0, cfg.n_regions, size=cfg.n_users)
    names = [f"segment_{k:02d}" for k in range(cfg.n_categories)] + [
        f"region_{r}" for r in range(cfg.n_regions)
    ]
    features = np.zeros((cfg.n_users, len(names)), dtype=np.float32)
    features[np.arange(cfg.n_users), user_segment] = 1.0
    features[np.arange(cfg.n_users), cfg.n_categories + region] = 1.0
    return UserTable(user_ids=np.arange(cfg.n_users) + 1, features=features, feature_names=names)
