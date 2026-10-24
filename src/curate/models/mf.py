"""Matrix factorisation trained with the Bayesian Personalised Ranking loss (Rendle et al. 2009).

For each observed (user, item) pair a negative item the user has not interacted with is sampled; the
model learns to score the observed item above the negative. Training is plain PyTorch on CPU with a fixed
seed and a capped thread count so a run is reproducible and does not starve other processes.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import torch
from torch import nn

from curate.data.interactions import Interactions
from curate.models.base import FitContext


def unique_pairs(train: Interactions) -> tuple[np.ndarray, np.ndarray]:
    """Distinct (user, item) pairs in a deterministic order."""
    coo = train.to_csr().tocoo()
    order = np.lexsort((coo.col, coo.row))
    return coo.row[order].astype(np.int64), coo.col[order].astype(np.int64)


def sample_negatives(
    rng: np.random.Generator, users: np.ndarray, positive_keys: np.ndarray, n_items: int, max_rounds: int = 5
) -> np.ndarray:
    """Uniform negatives with a few rounds of rejection against the user's known positives."""
    negatives = rng.integers(0, n_items, size=len(users))
    for _ in range(max_rounds):
        clash = np.isin(users * n_items + negatives, positive_keys)
        if not clash.any():
            break
        negatives[clash] = rng.integers(0, n_items, size=int(clash.sum()))
    return negatives


class BPRRecommender:
    """Dot-product factorisation with item biases, trained on the BPR pairwise loss."""

    name = "bpr"

    def __init__(
        self,
        *,
        dim: int = 128,
        epochs: int = 50,
        lr: float = 0.005,
        reg: float = 3e-3,
        batch_size: int = 2048,
        seed: int = 42,
        n_threads: int = 4,
    ) -> None:
        self.dim, self.epochs, self.lr, self.reg = dim, epochs, lr, reg
        self.batch_size, self.seed, self.n_threads = batch_size, seed, n_threads
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.loss_history: list[float] = []

    def fit(self, train: Interactions, context: FitContext) -> Self:  # noqa: ARG002
        """Run ``epochs`` passes of mini-batch BPR with Adam."""
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.n_threads)
        rng = np.random.default_rng(self.seed)
        users, items = unique_pairs(train)
        keys = np.sort(users * train.n_items + items)
        user_emb = nn.Embedding(train.n_users, self.dim)
        item_emb = nn.Embedding(train.n_items, self.dim)
        bias = nn.Embedding(train.n_items, 1)
        for emb in (user_emb, item_emb):
            nn.init.normal_(emb.weight, std=0.1 / np.sqrt(self.dim))
        nn.init.zeros_(bias.weight)
        optimiser = torch.optim.Adam(
            [*user_emb.parameters(), *item_emb.parameters(), *bias.parameters()], lr=self.lr
        )
        self.loss_history = []
        for _ in range(self.epochs):
            order = rng.permutation(len(users))
            negatives = sample_negatives(rng, users, keys, train.n_items)
            total, n_batches = 0.0, 0
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                loss = self._batch_loss(user_emb, item_emb, bias, users[idx], items[idx], negatives[idx])
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total, n_batches = total + float(loss.detach()), n_batches + 1
            self.loss_history.append(total / max(n_batches, 1))
        self.user_factors = user_emb.weight.detach().numpy().astype(np.float32)
        self.item_factors = item_emb.weight.detach().numpy().astype(np.float32)
        self.item_bias = bias.weight.detach().numpy().ravel().astype(np.float32)
        return self

    def _batch_loss(
        self,
        user_emb: nn.Embedding,
        item_emb: nn.Embedding,
        bias: nn.Embedding,
        users: np.ndarray,
        positives: np.ndarray,
        negatives: np.ndarray,
    ) -> torch.Tensor:
        u = user_emb(torch.from_numpy(users))
        pos, neg = item_emb(torch.from_numpy(positives)), item_emb(torch.from_numpy(negatives))
        b_pos, b_neg = (
            bias(torch.from_numpy(positives)).squeeze(-1),
            bias(torch.from_numpy(negatives)).squeeze(-1),
        )
        x_ui = (u * pos).sum(-1) + b_pos
        x_uj = (u * neg).sum(-1) + b_neg
        penalty = self.reg * (u.pow(2).sum() + pos.pow(2).sum() + neg.pow(2).sum() + b_pos.pow(2).sum())
        return -nn.functional.logsigmoid(x_ui - x_uj).mean() + penalty / len(users)

    def score(self, users: np.ndarray) -> np.ndarray:
        """Dot products plus item bias."""
        if self.user_factors is None or self.item_factors is None or self.item_bias is None:
            raise RuntimeError("fit must be called before score")
        return (self.user_factors[users] @ self.item_factors.T + self.item_bias).astype(np.float64)
