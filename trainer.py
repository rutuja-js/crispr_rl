"""Training loop for bandit policies with learning curve tracking."""

from __future__ import annotations

import time
from typing import Literal

import numpy as np

from crispr_rl.rl.env import GuideRNAEnv
from crispr_rl.rl.bandit import make_bandit, EpsilonGreedyBandit, UCBBandit
from crispr_rl.utils.seeds import set_global_seed
from crispr_rl.utils.logging import get_logger


class BanditTrainer:
    """
    Training loop that runs a bandit policy on the GuideRNAEnv.

    Tracks rewards per step, learning curves, and best candidates found.
    """

    def __init__(
        self,
        candidates: list[dict],
        sequence: str,
        weights: dict | None = None,
        constraints: dict | None = None,
        policy: Literal["epsilon_greedy", "ucb"] = "epsilon_greedy",
        rl_config: dict | None = None,
        seed: int = 42,
        run_id: str | None = None,
    ):
        self.candidates = candidates
        self.sequence = sequence
        self.weights = weights
        self.constraints = constraints
        self.policy = policy
        self.rl_config = rl_config or {}
        self.seed = seed
        self.run_id = run_id or "rl_run"
        self.logger = get_logger(__name__, run_id=self.run_id)

        set_global_seed(seed)

        self.env = GuideRNAEnv(
            candidates=candidates,
            sequence=sequence,
            weights=weights,
            constraints=constraints,
            seed=seed,
        )
        self.bandit = make_bandit(
            policy=policy,
            n_arms=len(candidates),
            config=rl_config,
            seed=seed,
        )

        # Tracking
        self.reward_history: list[float] = []
        self.best_reward = -float("inf")
        self.best_candidate: dict | None = None
        self.n_steps = 0

    def train(self, n_steps: int = 500) -> dict:
        """
        Run the training loop for *n_steps* steps.

        Returns a summary dict with learning curves and best result.
        """
        start = time.perf_counter()
        state = self.env.reset()
        cumulative_reward = 0.0

        for step in range(n_steps):
            action = self.bandit.select_arm(context=state)
            next_state, reward, done, info = self.env.step(action)
            self.bandit.update(action, reward)

            self.reward_history.append(reward)
            cumulative_reward += reward

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_candidate = self.candidates[action].copy()
                self.best_candidate["reward"] = reward

            if done:
                state = self.env.reset()
            else:
                state = next_state

            self.n_steps += 1

        elapsed = time.perf_counter() - start
        self.logger.info(
            f"Training complete: {n_steps} steps in {elapsed:.2f}s, "
            f"best_reward={self.best_reward:.4f}"
        )

        return self._build_summary(elapsed)

    def _build_summary(self, elapsed: float) -> dict:
        rewards = np.array(self.reward_history)
        return {
            "n_steps": self.n_steps,
            "policy": self.policy,
            "seed": self.seed,
            "best_reward": float(self.best_reward),
            "best_candidate": self.best_candidate,
            "mean_reward": float(np.mean(rewards)) if len(rewards) > 0 else 0.0,
            "std_reward": float(np.std(rewards)) if len(rewards) > 0 else 0.0,
            "learning_curve": rewards.tolist(),
            "latency_ms": round(elapsed * 1000, 2),
            "bandit_state": self.bandit.get_state(),
        }

    def top_k_candidates(self, k: int = 5) -> list[dict]:
        """Return top-k candidates sorted by their last observed RL reward."""
        scored = []
        for i, cand in enumerate(self.candidates):
            reward = self.bandit.values[i] if hasattr(self.bandit, "values") else cand.get("reward", 0.0)
            scored.append((reward, cand))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]
