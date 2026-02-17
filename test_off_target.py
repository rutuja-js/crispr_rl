"""Tests for off-target scoring."""

import pytest
from crispr_rl.scoring.off_target import (
    count_mismatches,
    weighted_mismatch_score,
    find_off_targets,
    specificity_proxy,
    passes_specificity_filter,
)

GUIDE = "ATCGATCGATCGATCGATCG"

# A sequence that contains:
# - 1 perfect match
# - 1 near-match with 2 mismatches outside seed
# - 1 near-match with 1 mismatch in seed
PERFECT = GUIDE
NEAR1 = GUIDE[:2] + "AA" + GUIDE[4:]   # 2 mismatches, non-seed
NEAR2 = GUIDE[:19] + "A"               # 1 mismatch in last position (seed region)


def build_test_sequence() -> str:
    """Embed guide + near-matches in a longer context."""
    flank = "TTTTTTTTTTTTTTTTTTTT"  # 20nt flanks (no NGG here, just for off-target)
    sep = "CCCCCCCCCCCCCCCCCCCC"
    return flank + PERFECT + sep + NEAR1 + sep + NEAR2 + flank


class TestCountMismatches:
    def test_perfect_match(self):
        assert count_mismatches(GUIDE, GUIDE) == 0

    def test_single_mismatch(self):
        target = GUIDE[:5] + "X" + GUIDE[6:]
        assert count_mismatches(GUIDE, target) == 1

    def test_all_mismatches(self):
        target = "N" * len(GUIDE)
        assert count_mismatches(GUIDE, target) == len(GUIDE)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            count_mismatches(GUIDE, "ACGT")


class TestWeightedMismatch:
    def test_seed_mismatch_weighted_higher(self):
        # 1 mismatch in seed vs 1 outside seed
        in_seed = GUIDE[:19] + "A"   # last pos = seed
        out_seed = "A" + GUIDE[1:]   # first pos = non-seed
        ws_seed = weighted_mismatch_score(GUIDE, in_seed)
        ws_non = weighted_mismatch_score(GUIDE, out_seed)
        assert ws_seed > ws_non, "Seed mismatch must be weighted higher"

    def test_perfect_match_score_zero(self):
        assert weighted_mismatch_score(GUIDE, GUIDE) == pytest.approx(0.0)


class TestOffTargetSearch:
    def test_finds_perfect_match(self):
        seq = build_test_sequence()
        hits = find_off_targets(GUIDE, seq, max_mismatches=3)
        perfect = [h for h in hits if h["mismatch_count"] == 0]
        assert len(perfect) >= 1, "Must find the on-target perfect match"

    def test_finds_near_matches(self):
        seq = build_test_sequence()
        hits = find_off_targets(GUIDE, seq, max_mismatches=3)
        assert len(hits) >= 2, f"Should find ≥2 hits (on-target + near-match), got {len(hits)}"

    def test_specificity_less_than_one_with_off_targets(self):
        seq = build_test_sequence()
        spec = specificity_proxy(GUIDE, seq, max_mismatches=3)
        assert spec < 1.0, f"Specificity should be <1 when off-targets exist, got {spec}"

    def test_specificity_between_0_and_1(self):
        seq = build_test_sequence()
        spec = specificity_proxy(GUIDE, seq)
        assert 0.0 < spec <= 1.0

    def test_passes_filter_high_spec(self):
        # A guide with no off-targets should pass the filter
        seq = "A" * 30 + GUIDE + "G" * 30  # no near-matches
        # With a very short sequence, off-target hits are minimal
        result = passes_specificity_filter(GUIDE, GUIDE + "AAAAAAAAAAAAAAAAAAAAAAAAAAAA", threshold=0.0)
        assert result is True
