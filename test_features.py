"""Tests for feature extraction modules."""

import pytest
from crispr_rl.features.gc_content import (
    gc_fraction,
    is_gc_optimal,
    max_homopolymer_run,
    run_length_penalty,
    gc_penalty,
)
from crispr_rl.features.thermo_proxy import tm_wallace, tm_nearest_neighbour, efficiency_proxy
from crispr_rl.features.context import extract_context_window, positional_score


class TestGCContent:
    def test_pure_gc(self):
        assert gc_fraction("GCGCGCGC") == pytest.approx(1.0)

    def test_pure_at(self):
        assert gc_fraction("ATATAT") == pytest.approx(0.0)

    def test_mixed(self):
        assert gc_fraction("GCTA") == pytest.approx(0.5)

    def test_empty(self):
        assert gc_fraction("") == pytest.approx(0.0)

    def test_case_insensitive(self):
        assert gc_fraction("gcgc") == pytest.approx(1.0)

    def test_optimal_range(self):
        assert is_gc_optimal(0.5) is True
        assert is_gc_optimal(0.3) is False
        assert is_gc_optimal(0.7) is False

    def test_gc_penalty_in_range(self):
        assert gc_penalty(0.5) == pytest.approx(0.0)

    def test_gc_penalty_below(self):
        pen = gc_penalty(0.3)
        assert pen > 0.0
        assert pen == pytest.approx(0.4 - 0.3)


class TestHomopolymer:
    def test_no_run(self):
        assert max_homopolymer_run("ACGT") == 1

    def test_run_of_4(self):
        assert max_homopolymer_run("ACGTAAAACGT") == 4

    def test_run_of_6(self):
        assert max_homopolymer_run("GGGGGG") == 6

    def test_penalty_below_threshold(self):
        assert run_length_penalty("ACGT", threshold=4) == pytest.approx(0.0)

    def test_penalty_above_threshold(self):
        # 6-base run, threshold=4 → excess=2 → penalty=0.4
        pen = run_length_penalty("AAAAAA", threshold=4)
        assert pen > 0.0

    def test_penalty_maxes_at_1(self):
        # Very long run
        pen = run_length_penalty("A" * 100, threshold=4)
        assert pen <= 1.0


class TestThermo:
    def test_tm_wallace_all_gc(self):
        seq = "GCGCGCGCGCGCGCGCGCGC"  # 20nt all GC
        tm = tm_wallace(seq)
        assert tm == pytest.approx(4 * 20)  # 80

    def test_tm_wallace_all_at(self):
        seq = "ATATATAT"
        tm = tm_wallace(seq)
        assert tm == pytest.approx(2 * 8)

    def test_tm_nearest_neighbour_returns_float(self):
        seq = "ATCGATCGATCGATCGATCG"
        tm = tm_nearest_neighbour(seq)
        assert isinstance(tm, float)
        assert 20 < tm < 100  # reasonable range

    def test_efficiency_proxy_between_0_and_1(self):
        score = efficiency_proxy("ATCGATCGATCGATCGATCG")
        assert 0.0 <= score <= 1.0


class TestContext:
    def test_extract_context_returns_dict(self):
        seq = "A" * 100
        ctx = extract_context_window(seq, position=30)
        assert "upstream" in ctx
        assert "guide" in ctx
        assert "downstream" in ctx

    def test_guide_length_in_context(self):
        seq = "A" * 100
        ctx = extract_context_window(seq, position=30, guide_length=20)
        assert len(ctx["guide"]) == 20

    def test_positional_score_near_tss(self):
        score = positional_score(10, 1000, preferred_frac=0.2)
        assert score == pytest.approx(1.0)

    def test_positional_score_far_from_tss(self):
        score = positional_score(900, 1000, preferred_frac=0.2)
        assert score < 0.5
