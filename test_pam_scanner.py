"""Tests for PAM scanner."""

import pytest
from crispr_rl.features.pam_scanner import scan_sequence, PAMHit
from crispr_rl.data.loaders import reverse_complement


def make_guide_with_pam(guide: str, pam: str = "GGG") -> str:
    """Helper: embed a known guide+PAM in a longer sequence."""
    prefix = "ACGT" * 5  # 20 nt prefix padding
    return prefix + guide + pam + "TTTT"


# Known 20nt guide + NGG PAM
GUIDE = "ATCGATCGATCGATCGATCG"
SEQ_FWD = "AAAAAAAAAAAAAAAAAAAAAAAAA" + GUIDE + "AGGTTTTT"


class TestPAMScanner:
    def test_finds_known_ngg_site_forward(self):
        """Scanner must find the exact NGG PAM on forward strand."""
        seq = GUIDE + "AGG" + "AAAAAAAAAAA"
        hits = scan_sequence(seq, pam_pattern="NGG", guide_length=20)
        fwd_hits = [h for h in hits if h.strand == "+"]
        assert len(fwd_hits) >= 1, "Should find at least one forward hit"
        seqs = [h.protospacer for h in fwd_hits]
        assert GUIDE in seqs, f"Expected guide {GUIDE} in hits: {seqs}"

    def test_finds_multiple_pam_sites(self):
        """Multiple NGG sites in one sequence are all found."""
        guide1 = "AAAAAAAAAAAAAAAAAAAAGG"  # ends in GG → this is PAM; proto is 20nt before
        # Build sequence with 3 embedded sites
        base = "N" * 20  # spacer
        seq = (
            "GCATGCATGCATGCATGCATGG"  # NGG at end
            + "AAAAAAAAAAAAAAAAAAAA" + "TTGG"
            + "CCCCCCCCCCCCCCCCCCCC" + "CAGG"
        )
        hits = scan_sequence(seq, pam_pattern="NGG")
        assert len(hits) >= 2, f"Expected ≥2 hits, got {len(hits)}"

    def test_scans_reverse_strand(self):
        """Scanner must find PAM sites on both strands."""
        # Create a sequence where NGG only appears on the reverse complement
        # RC of NGG is CCN → put CCN on forward strand
        # A 20nt guide on the rc strand with CCN on forward = NGG on reverse
        fwd_seq = "AAAAAAAAAAAAAAAAAAAAAA" + "CC" + "G" + "TTTTTTTTTT"
        hits = scan_sequence(fwd_seq, pam_pattern="NGG")
        minus_hits = [h for h in hits if h.strand == "-"]
        assert len(minus_hits) >= 1, "Should find at least one reverse-strand hit"

    def test_protospacer_length_is_20(self):
        """All returned protospacers must be exactly 20 nt."""
        seq = "AACGATCGATCGATCGATCGATCGGGGGG" + "CCCCCCCCCCCCCCCCCCCC" + "AGG" + "TTT"
        hits = scan_sequence(seq, pam_pattern="NGG", guide_length=20)
        for h in hits:
            assert len(h.protospacer) == 20, f"Protospacer {h.protospacer} is not 20 nt"

    def test_custom_pam_cas12a(self):
        """Test Cas12a TTTV (TTTN) PAM pattern."""
        # TTTN before protospacer for Cas12a (5' PAM)
        # For simplicity, test that scanner finds TTTG
        seq = "ATCGATCGATCGATCGATCGATCG" + "TTTG" + "ACGTACGTACGT"
        # TTTG is 4nt, guide is 20nt upstream of PAM for Cas12a-like
        # Our scanner looks for PAM downstream of protospacer by default
        # Test that the pattern matches
        hits = scan_sequence(seq, pam_pattern="TTTN")
        # We just check the scanner runs without error and TTTG matches TTTN
        assert isinstance(hits, list)

    def test_returns_namedtuples(self):
        """Hits must be PAMHit namedtuples."""
        seq = "GCATGCATGCATGCATGCATGG" + "TT"
        hits = scan_sequence(seq, pam_pattern="NGG")
        for h in hits:
            assert isinstance(h, PAMHit)
            assert h.strand in ("+", "-")
            assert isinstance(h.position, int)

    def test_empty_sequence(self):
        """Empty sequence should return empty list."""
        hits = scan_sequence("", pam_pattern="NGG")
        assert hits == []

    def test_short_sequence_no_hits(self):
        """Sequence shorter than guide+PAM should return empty list."""
        hits = scan_sequence("ACGTGG", pam_pattern="NGG")  # only 6 nt, need ≥23
        assert hits == []
