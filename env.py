"""Gym-like RL environment for guide RNA selection."""

from __future__ import annotations

import numpy as np
from typing import Any

from crispr_rl.features.gc_content import gc_fraction, max_homopolymer_run
from crispr_rl.features.thermo_proxy import efficiency_proxy
from crispr_rl.scoring.composite import composite_reward
from crispr_rl.scoring.off_target import specificity_proxy


class GuideRNAEnv:
    """
    Gym-compatible environment for guide RNA selection.

    State:  flattened feature vector per candidate
    Action: index into candidate pool
    Reward: composite score from scoring/composite.py
    """

    def __init__(
        self,
        candidates: list[dict],
        sequence: str,
        weights: dict[str, float] | None = None,
        constraints: dict | None = None,
        seed: int = 42,
    ):
        if not candidates:
            raise ValueError("Candidate pool must not be empty.")
        self.candidates = candidates
        self.sequence = sequence
        self.weights = weights or {"w_efficiency": 0.5, "w_specificity": 0.3, "w_coverage": 0.2}
        self.constraints = constraints or {}
        self.rng = np.random.default_rng(seed)
        self.n_actions = len(candidates)
        self.action_space = list(range(self.n_actions))
        self._current_step = 0
        self._selected: list[int] = []
        self._state: np.ndarray | None = None
        self.reset()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, candidate: dict) -> np.ndarray:
        """Build a 5-dimensional feature vector for a single candidate."""
        seq = candidate["seq"]
        pos = candidate.get("position", 0)
        seq_len = max(1, len(self.sequence))

        gc = gc_fraction(seq)
        thermo = efficiency_proxy(seq)
        spec = specificity_proxy(seq, self.sequence)
        hp = max_homopolymer_run(seq) / 20.0  # normalised
        loc = pos / seq_len  # normalised position

        return np.array([gc, thermo, spec, hp, loc], dtype=np.float32)

    def _build_state(self) -> np.ndarray:
        """Concatenate feature vectors for all candidates → flat state."""
        feat_list = [self._extract_features(c) for c in self.candidates]
        return np.concatenate(feat_list).astype(np.float32)

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset environment; returns initial state vector."""
        self._current_step = 0
        self._selected = []
        self._state = self._build_state()
        return self._state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Take an action (select a guide by index).

        Returns (next_state, reward, done, info).
        """
        if action not in self.action_space:
            raise ValueError(f"Action {action} out of range [0, {self.n_actions})")

        candidate = self.candidates[action]
        reward, components = composite_reward(
            guide_seq=candidate["seq"],
            sequence=self.sequence,
            position=candidate.get("position", 0),
            weights=self.weights,
            constraints=self.constraints,
        )

        # Update candidate with RL reward
        candidate["reward"] = reward
        candidate["components"] = components

        self._selected.append(action)
        self._current_step += 1
        done = self._current_step >= self.n_actions
        info = {
            "action": action,
            "candidate_id": candidate.get("id", str(action)),
            "components": components,
            "step": self._current_step,
        }
        return self._state.copy(), float(reward), done, info

    def render(self, mode: str = "text") -> str:
        """Return a text summary of the environment state."""
        lines = [f"GuideRNAEnv — {self.n_actions} candidates, step {self._current_step}"]
        for i, c in enumerate(self.candidates):
            marker = "→" if i in self._selected else " "
            score = c.get("reward", c.get("score", 0.0))
            lines.append(f"  {marker} [{i:02d}] {c.get('id','?')} seq={c['seq'][:10]}… score={score:.4f}")
        return "\n".join(lines)

    @property
    def observation_space_shape(self) -> tuple[int]:
        return (self.n_actions * 5,)
