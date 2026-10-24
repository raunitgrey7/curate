"""Two-tower retriever: a user tower and an item tower trained with in-batch sampled softmax.

Each tower concatenates an id embedding with side features from the feature store and passes them through
a small MLP; the output is L2 normalised so scores are cosine similarities divided by a temperature. In-batch
negatives are corrected for their sampling frequency with the logQ term of Yi et al. (2019) so popular items
are not over-penalised. Items that appear twice in a batch are masked so a duplicate positive is never
treated as a negative.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import torch
from torch import nn

from curate.data.interactions import Interactions
from curate.models.base import FitContext
from curate.models.mf import unique_pairs


class Tower(nn.Module):
    """Id embedding concatenated with dense features, then a two-layer MLP with unit-norm output."""

    def __init__(self, n_entities: int, n_features: int, id_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.id_embedding = nn.Embedding(n_entities, id_dim)
        nn.init.normal_(self.id_embedding.weight, std=0.05)
        self.mlp = nn.Sequential(
            nn.Linear(id_dim + n_features, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, ids: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Return unit-norm embeddings for the given entity ids."""
        x = torch.cat([self.id_embedding(ids), features[ids]], dim=-1)
        return nn.functional.normalize(self.mlp(x), dim=-1)


class TwoTowerRecommender:
    """Retrieve by cosine similarity between learned user and item embeddings."""

    name = "two_tower"

    def __init__(
        self,
        *,
        dim: int = 64,
        id_dim: int = 32,
        hidden: int = 128,
        epochs: int = 20,
        batch_size: int = 1024,
        lr: float = 0.003,
        temperature: float = 0.1,
        logq_correction: bool = True,
        seed: int = 42,
        n_threads: int = 4,
    ) -> None:
        self.dim, self.id_dim, self.hidden = dim, id_dim, hidden
        self.epochs, self.batch_size, self.lr, self.temperature = epochs, batch_size, lr, temperature
        self.logq_correction, self.seed, self.n_threads = logq_correction, seed, n_threads
        self.user_embeddings: np.ndarray | None = None
        self.item_embeddings: np.ndarray | None = None
        self.loss_history: list[float] = []

    def fit(self, train: Interactions, context: FitContext) -> Self:
        """Train both towers and cache the resulting embeddings for every known user and item."""
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.n_threads)
        rng = np.random.default_rng(self.seed)
        users, items = unique_pairs(train)
        user_feats = torch.from_numpy(np.asarray(context.user_features, dtype=np.float32))
        item_feats = torch.from_numpy(np.asarray(context.item_features, dtype=np.float32))
        user_tower = Tower(train.n_users, user_feats.shape[1], self.id_dim, self.hidden, self.dim)
        item_tower = Tower(train.n_items, item_feats.shape[1], self.id_dim, self.hidden, self.dim)
        freq = np.bincount(items, minlength=train.n_items).astype(np.float64)
        log_q = torch.from_numpy(np.log(freq / freq.sum() + 1e-12).astype(np.float32))
        optimiser = torch.optim.Adam([*user_tower.parameters(), *item_tower.parameters()], lr=self.lr)
        self.loss_history = []
        for _ in range(self.epochs):
            order = rng.permutation(len(users))
            total, n_batches = 0.0, 0
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                batch_users, batch_items = torch.from_numpy(users[idx]), torch.from_numpy(items[idx])
                u = user_tower(batch_users, user_feats)
                v = item_tower(batch_items, item_feats)
                loss = self._batch_loss(u, v, batch_items, log_q)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total, n_batches = total + float(loss.detach()), n_batches + 1
            self.loss_history.append(total / max(n_batches, 1))
        with torch.no_grad():
            self.user_embeddings = (
                user_tower(torch.arange(train.n_users), user_feats).numpy().astype(np.float32)
            )
            self.item_embeddings = (
                item_tower(torch.arange(train.n_items), item_feats).numpy().astype(np.float32)
            )
        return self

    def _batch_loss(
        self, u: torch.Tensor, v: torch.Tensor, items: torch.Tensor, log_q: torch.Tensor
    ) -> torch.Tensor:
        logits = (u @ v.T) / self.temperature
        if self.logq_correction:
            logits = logits - log_q[items][None, :]
        duplicate = items[:, None] == items[None, :]
        duplicate.fill_diagonal_(False)
        logits = logits.masked_fill(duplicate, float("-inf"))
        return nn.functional.cross_entropy(logits, torch.arange(len(items)))

    def score(self, users: np.ndarray) -> np.ndarray:
        """Cosine similarity between user and item embeddings."""
        if self.user_embeddings is None or self.item_embeddings is None:
            raise RuntimeError("fit must be called before score")
        return (self.user_embeddings[users] @ self.item_embeddings.T).astype(np.float64)
