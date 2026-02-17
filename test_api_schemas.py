"""Tests for Pydantic API schemas."""

import pytest
from pydantic import ValidationError

from crispr_rl.api.schemas import (
    GuideRNACandidate,
    DesignRequest,
    FeedbackRequest,
    RLScore,
)


class TestGuideRNACandidate:
    def _valid(self, **kwargs):
        base = {
            "id": "g001",
            "seq": "ATCGATCGATCGATCGATCG",
            "pam": "AGG",
            "locus": "chr1:100-120",
            "strand": "+",
            "gc": 0.5,
        }
        base.update(kwargs)
        return GuideRNACandidate(**base)

    def test_valid_candidate(self):
        c = self._valid()
        assert c.seq == "ATCGATCGATCGATCGATCG"
        assert c.gc == 0.5

    def test_invalid_seq_length(self):
        with pytest.raises(ValidationError):
            self._valid(seq="ACGT")  # only 4 nt

    def test_invalid_seq_characters(self):
        with pytest.raises(ValidationError):
            self._valid(seq="ATCGATCGATCGATCGATCX")  # X is invalid

    def test_invalid_gc_out_of_range(self):
        with pytest.raises(ValidationError):
            self._valid(gc=1.5)

    def test_invalid_strand(self):
        with pytest.raises(ValidationError):
            self._valid(strand="?")

    def test_lowercase_seq_normalised(self):
        c = self._valid(seq="atcgatcgatcgatcgatcg")
        assert c.seq == "ATCGATCGATCGATCGATCG"

    def test_defaults(self):
        c = self._valid()
        assert c.score == 0.0
        assert c.risk_flags == []
        assert c.explanations == {}


class TestDesignRequest:
    def test_valid_with_sequence(self):
        r = DesignRequest(sequence="ATCGATCGATCGATCGATCGATCGATCGATCG", gene_ids=[])
        assert r.sequence is not None
        assert r.profile == "knockout"

    def test_valid_with_gene_ids(self):
        r = DesignRequest(gene_ids=["BRCA1", "TP53"])
        assert len(r.gene_ids) == 2

    def test_sequence_too_short(self):
        with pytest.raises(ValidationError):
            DesignRequest(sequence="ACGT")

    def test_invalid_profile(self):
        with pytest.raises(ValidationError):
            DesignRequest(gene_ids=["BRCA1"], profile="invalid")

    def test_invalid_sequence_chars(self):
        with pytest.raises(ValidationError):
            DesignRequest(sequence="ATCGATCGATCGATCGATCGATCGATCGATCX")

    def test_default_request_id_generated(self):
        r1 = DesignRequest(gene_ids=["BRCA1"])
        r2 = DesignRequest(gene_ids=["BRCA1"])
        assert r1.request_id != r2.request_id  # UUIDs should differ

    def test_seed_default(self):
        r = DesignRequest(gene_ids=["BRCA1"])
        assert r.seed == 42


class TestFeedbackRequest:
    def test_valid_feedback(self):
        fb = FeedbackRequest(candidate_id="g001", rating=5)
        assert fb.rating == 5

    def test_rating_below_1_fails(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(candidate_id="g001", rating=0)

    def test_rating_above_5_fails(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(candidate_id="g001", rating=6)

    def test_valid_boundary_ratings(self):
        for r in (1, 2, 3, 4, 5):
            fb = FeedbackRequest(candidate_id="g001", rating=r)
            assert fb.rating == r

    def test_optional_notes(self):
        fb = FeedbackRequest(candidate_id="g001", rating=3, notes="looks ok", rationale="prior lit")
        assert fb.notes == "looks ok"


class TestRLScore:
    def test_valid(self):
        s = RLScore(candidate_id="g001", reward=0.75, components={"efficiency": 0.8, "specificity": 0.7})
        assert s.reward == pytest.approx(0.75)
