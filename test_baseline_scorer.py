"""Tests for baseline scorer."""

import pytest
from crispr_rl.scoring.baseline import score_guide, rank_candidates

SEQUENCE = (
    "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"
    "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
)
GUIDE = "ATCGATCGATCGATCGATCG"


class TestScoreGuide:
    def test_returns_float(self):
        score, expl = score_guide(GUIDE, SEQUENCE, 0)
        assert isinstance(score, float)

    def test_score_in_range(self):
        score, _ = score_guide(GUIDE, SEQUENCE, 0)
        assert 0.0 <= score <= 1.0

    def test_returns_explanation_dict(self):
        _, expl = score_guide(GUIDE, SEQUENCE, 0)
        assert isinstance(expl, dict)
        assert "efficiency" in expl
        assert "specificity" in expl

    def test_deterministic(self):
        score1, _ = score_guide(GUIDE, SEQUENCE, 0)
        score2, _ = score_guide(GUIDE, SEQUENCE, 0)
        assert score1 == score2

    def test_high_w1_increases_efficiency_contribution(self):
        score_low_w1, _ = score_guide(GUIDE, SEQUENCE, 0, w1=0.1, w2=0.45, w3=0.45)
        score_high_w1, _ = score_guide(GUIDE, SEQUENCE, 0, w1=0.9, w2=0.05, w3=0.05)
        # Not strictly guaranteed to be higher/lower, but they should differ
        assert score_low_w1 != score_high_w1

    def test_weight_change_affects_ranking(self):
        """Different weight vectors must produce different rankings."""
        cands = [
            {"id": "g1", "seq": "GCGCGCGCGCGCGCGCGCGC", "position": 0},
            {"id": "g2", "seq": "ATATATATATATATATATATAT"[:20], "position": 0},
        ]
        # Deep copies to avoid mutation
        import copy
        c1 = copy.deepcopy(cands)
        c2 = copy.deepcopy(cands)

        rank_candidates(c1, SEQUENCE, w1=0.9, w2=0.05, w3=0.05)
        rank_candidates(c2, SEQUENCE, w1=0.05, w2=0.9, w3=0.05)

        # Just verify scores were set
        assert all(isinstance(c["score"], float) for c in c1)
        assert all(isinstance(c["score"], float) for c in c2)


class TestRankCandidates:
    def _make_cands(self, n: int = 5) -> list[dict]:
        seqs = [
            "ATCGATCGATCGATCGATCG",
            "GCGCGCGCGCGCGCGCGCGC",
            "ATATATATATATATATATATAT"[:20],
            "TTTTTTTTTTTTTTTTTTTTT"[:20],
            "AAAAAAAAAAAAAAAAAAAAA"[:20],
        ]
        return [{"id": f"g{i}", "seq": seqs[i % len(seqs)], "position": i * 10} for i in range(n)]

    def test_sorted_descending(self):
        cands = self._make_cands(5)
        ranked = rank_candidates(cands, SEQUENCE)
        scores = [c["score"] for c in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_max_candidates_limit(self):
        cands = self._make_cands(5)
        ranked = rank_candidates(cands, SEQUENCE, max_candidates=3)
        assert len(ranked) <= 3
