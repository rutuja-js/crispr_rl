"""Contextual bandit: epsilon-greedy and UCB policies."""

from __future__ import annotations

import math
import numpy as np
from typing import Literal


class EpsilonGreedyBandit:
    """
    Epsilon-greedy contextual bandit with exponential decay.

    Maintains per-arm running statistics (mean reward, pull count).
    """

    def __init__(
        self,
        n_arms: int,
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ):
        self.n_arms = n_arms
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        self.counts = np.zeros(n_arms, dtype=np.int32)
        self.values = np.zeros(n_arms, dtype=np.float64)  # running mean rewards

    def select_arm(self, context: np.ndarray | None = None) -> int:
        """Select an arm (greedy or explore) given optional context."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_arms))
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float) -> None:
        """Incremental update of running mean for arm."""
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def reset(self) -> None:
        self.counts[:] = 0
        self.values[:] = 0.0

    @property
    def best_arm(self) -> int:
        return int(np.argmax(self.values))

    def get_state(self) -> dict:
        return {
            "type": "epsilon_greedy",
            "epsilon": self.epsilon,
            "counts": self.counts.tolist(),
            "values": self.values.tolist(),
        }


class UCBBandit:
    """
    Upper Confidence Bound (UCB1) bandit policy.

    UCB score = mean_reward + c * sqrt(log(t) / n_pulls)
    """

    def __init__(
        self,
        n_arms: int,
        c: float = 1.414,
        seed: int = 42,
    ):
        self.n_arms = n_arms
        self.c = c
        self.rng = np.random.default_rng(seed)

        self.counts = np.zeros(n_arms, dtype=np.int32)
        self.values = np.zeros(n_arms, dtype=np.float64)
        self.t = 0  # total pulls

    def select_arm(self, context: np.ndarray | None = None) -> int:
        """Select arm with highest UCB score. Unpulled arms have infinite UCB."""
        self.t += 1
        # Pull any unpulled arm first (round-robin initialisation)
        unpulled = np.where(self.counts == 0)[0]
        if len(unpulled) > 0:
            return int(unpulled[0])

        ucb_scores = self.values + self.c * np.sqrt(
            np.log(self.t) / np.maximum(self.counts, 1)
        )
        return int(np.argmax(ucb_scores))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def reset(self) -> None:
        self.counts[:] = 0
        self.values[:] = 0.0
        self.t = 0

    @property
    def best_arm(self) -> int:
        return int(np.argmax(self.values))

    def get_state(self) -> dict:
        return {
            "type": "ucb",
            "c": self.c,
            "t": self.t,
            "counts": self.counts.tolist(),
            "values": self.values.tolist(),
        }


def make_bandit(
    policy: Literal["epsilon_greedy", "ucb"],
    n_arms: int,
    config: dict | None = None,
    seed: int = 42,
) -> EpsilonGreedyBandit | UCBBandit:
    """Factory function for creating bandit instances from config."""
    cfg = config or {}
    if policy == "epsilon_greedy":
        return EpsilonGreedyBandit(
            n_arms=n_arms,
            epsilon_start=cfg.get("epsilon_start", 0.3),
            epsilon_end=cfg.get("epsilon_end", 0.05),
            epsilon_decay=cfg.get("epsilon_decay", 0.995),
            seed=seed,
        )
    elif policy == "ucb":
        return UCBBandit(
            n_arms=n_arms,
            c=cfg.get("ucb_c", 1.414),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown policy '{policy}'. Choose 'epsilon_greedy' or 'ucb'.")
