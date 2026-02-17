"""Tests for the RL environment and bandit policies."""

import pytest
import numpy as np
from crispr_rl.rl.env import GuideRNAEnv
from crispr_rl.rl.bandit import EpsilonGreedyBandit, UCBBandit, make_bandit

SEQUENCE = (
    "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"
    "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
)

CANDIDATES = [
    {"id": "g0", "seq": "ATCGATCGATCGATCGATCG", "pam": "AGG", "locus": "pos:0", "strand": "+", "gc": 0.5, "position": 0},
    {"id": "g1", "seq": "GCTAGCTAGCTAGCTAGCTA", "pam": "TGG", "locus": "pos:20", "strand": "+", "gc": 0.5, "position": 20},
    {"id": "g2", "seq": "TATATATATATATATATATA", "pam": "CGG", "locus": "pos:40", "strand": "-", "gc": 0.0, "position": 40},
]


class TestGuideRNAEnv:
    def test_reset_returns_array(self):
        env = GuideRNAEnv(CANDIDATES, SEQUENCE, seed=42)
        state = env.reset()
        assert isinstance(state, np.ndarray)

    def test_state_shape(self):
        env = GuideRNAEnv(CANDIDATES, SEQUENCE, seed=42)
        state = env.reset()
        expected = len(CANDIDATES) * 5
        assert state.shape == (expected,)

    def test_step_returns_4_tuple(self):
        env = GuideRNAEnv(CANDIDATES, SEQUENCE, seed=42)
        env.reset()
        result = env.step(0)
        assert len(result) == 4

    def test_reward_is_float(self):
        env = GuideRNAEnv(CANDIDATES, SEQUENCE, seed=42)
        env.reset()
        _, reward, _, _ = env.step(0)
        assert isinstance(reward, float)

    def test_action_space_valid(self):
        env = GuideRNAEnv(CANDIDATES, SEQUENCE, seed=42)
        assert env.action_space == [0, 1, 2]
        assert env.n_actions == 3

    def test_invalid_action_raises(self):
        env = GuideRNAEnv(CANDIDATES, SEQUENCE, seed=42)
        env.reset()
        with pytest.raises(ValueError):
            env.step(99)

    def test_done_after_all_actions(self):
        env = GuideRNAEnv(CANDIDATES, SEQUENCE, seed=42)
        env.reset()
        done = False
        for i in range(len(CANDIDATES)):
            _, _, done, _ = env.step(i % len(CANDIDATES))
        assert done is True

    def test_render_returns_string(self):
        env = GuideRNAEnv(CANDIDATES, SEQUENCE, seed=42)
        env.reset()
        env.step(0)
        text = env.render()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_empty_candidates_raises(self):
        with pytest.raises(ValueError):
            GuideRNAEnv([], SEQUENCE)


class TestBandits:
    def test_epsilon_greedy_select_arm_in_range(self):
        bandit = EpsilonGreedyBandit(n_arms=5, seed=42)
        for _ in range(50):
            arm = bandit.select_arm()
            assert 0 <= arm < 5

    def test_epsilon_greedy_update(self):
        bandit = EpsilonGreedyBandit(n_arms=3, seed=42)
        bandit.update(0, 0.9)
        assert bandit.counts[0] == 1
        assert bandit.values[0] == pytest.approx(0.9)

    def test_epsilon_decays(self):
        bandit = EpsilonGreedyBandit(n_arms=3, epsilon_start=0.3, epsilon_decay=0.5, seed=42)
        eps_start = bandit.epsilon
        bandit.update(0, 0.5)
        assert bandit.epsilon < eps_start

    def test_ucb_select_arm_in_range(self):
        bandit = UCBBandit(n_arms=5, seed=42)
        for _ in range(20):
            arm = bandit.select_arm()
            assert 0 <= arm < 5

    def test_ucb_explores_all_arms_first(self):
        """UCB should pull each arm once before exploiting."""
        bandit = UCBBandit(n_arms=3, seed=42)
        arms = set()
        for _ in range(3):
            arm = bandit.select_arm()
            bandit.update(arm, 0.5)
            arms.add(arm)
        assert arms == {0, 1, 2}

    def test_make_bandit_factory(self):
        b = make_bandit("epsilon_greedy", n_arms=4, seed=0)
        assert isinstance(b, EpsilonGreedyBandit)
        b2 = make_bandit("ucb", n_arms=4, seed=0)
        assert isinstance(b2, UCBBandit)

    def test_make_bandit_invalid_policy(self):
        with pytest.raises(ValueError):
            make_bandit("unknown_policy", n_arms=3)
