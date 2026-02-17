"""Pydantic schemas for the crispr_rl FastAPI service.

Falls back to dataclasses when pydantic is not available.
"""
from __future__ import annotations

try:
    from pydantic import BaseModel, Field, field_validator
    _PYDANTIC = True
except ImportError:
    _PYDANTIC = False

import uuid
from typing import Literal

# ---------------------------------------------------------------------------
# Pure-dataclass fallback (no pydantic required for core logic)
# ---------------------------------------------------------------------------

if _PYDANTIC:
    from typing import Optional

    class GuideRNACandidate(BaseModel):
        id: str
        seq: str
        pam: str
        locus: str
        strand: Literal["+", "-"]
        gc: float
        features: dict = {}
        score: float = 0.0
        explanations: dict = {}
        risk_flags: list[str] = []

        @field_validator("seq")
        @classmethod
        def validate_seq(cls, v: str) -> str:
            v = v.upper().strip()
            invalid = set(v) - set("ACGTN")
            if invalid:
                raise ValueError(f"Guide sequence contains invalid characters: {invalid}")
            if len(v) != 20:
                raise ValueError(f"Guide sequence must be exactly 20 nt, got {len(v)}")
            return v

        @field_validator("gc")
        @classmethod
        def validate_gc(cls, v: float) -> float:
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"GC fraction must be in [0, 1], got {v}")
            return v

    class RLScore(BaseModel):
        candidate_id: str
        reward: float
        components: dict

    class DesignRequest(BaseModel):
        request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        gene_ids: list[str] = []
        organism: str = "human"
        sequence: Optional[str] = None
        params: dict = {}
        profile: Literal["knockout", "knockdown", "screening"] = "knockout"
        seed: int = 42

        @field_validator("sequence")
        @classmethod
        def validate_sequence(cls, v: Optional[str]) -> Optional[str]:
            if v is not None:
                v = v.upper().strip()
                invalid = set(v) - set("ACGTN")
                if invalid:
                    raise ValueError(f"Sequence contains invalid characters: {invalid}")
                if len(v) < 23:
                    raise ValueError(
                        f"Sequence too short ({len(v)} nt). Need at least 23 nt (20 guide + 3 PAM)."
                    )
            return v

        @field_validator("gene_ids", mode="before")
        @classmethod
        def validate_gene_ids(cls, v):
            return [str(g).strip() for g in v if str(g).strip()]

    class DesignResponse(BaseModel):
        request_id: str
        run_id: str
        candidates: list[GuideRNACandidate]
        metadata: dict

    class FeedbackRequest(BaseModel):
        candidate_id: str
        rating: int = Field(ge=1, le=5)
        notes: str = ""
        rationale: str = ""

    class ConfigResponse(BaseModel):
        profile: str
        weights: dict
        constraints: dict
        rl_config: dict

    class ConfigUpdateRequest(BaseModel):
        profile: Optional[str] = None
        weights: Optional[dict] = None
        constraints: Optional[dict] = None

    class MetricsResponse(BaseModel):
        total_requests: int
        total_feedback: int
        avg_score: float
        failure_count: int
        avg_latency_ms: float
        latency_p95_ms: float
        reward_breakdown: dict

else:
    # ---------------------------------------------------------------------------
    # Dataclass fallback — lightweight, no validation
    # ---------------------------------------------------------------------------
    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass
    class GuideRNACandidate:
        id: str
        seq: str
        pam: str
        locus: str
        strand: str
        gc: float
        features: dict = field(default_factory=dict)
        score: float = 0.0
        explanations: dict = field(default_factory=dict)
        risk_flags: list = field(default_factory=list)

        def __post_init__(self):
            self.seq = self.seq.upper().strip()
            invalid = set(self.seq) - set("ACGTN")
            if invalid:
                raise ValueError(f"Guide seq contains invalid chars: {invalid}")
            if len(self.seq) != 20:
                raise ValueError(f"Guide seq must be 20 nt, got {len(self.seq)}")
            if not (0.0 <= self.gc <= 1.0):
                raise ValueError(f"gc must be in [0,1], got {self.gc}")
            if self.strand not in ("+", "-"):
                raise ValueError(f"strand must be '+' or '-', got {self.strand}")

        def model_dump(self):
            return self.__dict__.copy()

    @dataclass
    class RLScore:
        candidate_id: str
        reward: float
        components: dict = field(default_factory=dict)

    @dataclass
    class DesignRequest:
        gene_ids: list = field(default_factory=list)
        organism: str = "human"
        sequence: Optional[str] = None
        params: dict = field(default_factory=dict)
        profile: str = "knockout"
        seed: int = 42
        request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

        def __post_init__(self):
            if self.sequence is not None:
                self.sequence = self.sequence.upper().strip()
                invalid = set(self.sequence) - set("ACGTN")
                if invalid:
                    raise ValueError(f"Sequence contains invalid characters: {invalid}")
                if len(self.sequence) < 23:
                    raise ValueError(f"Sequence too short ({len(self.sequence)} nt)")
            if self.profile not in ("knockout", "knockdown", "screening"):
                raise ValueError(f"Unknown profile: {self.profile}")

    @dataclass
    class DesignResponse:
        request_id: str
        run_id: str
        candidates: list
        metadata: dict

    @dataclass
    class FeedbackRequest:
        candidate_id: str
        rating: int
        notes: str = ""
        rationale: str = ""

        def __post_init__(self):
            if not (1 <= self.rating <= 5):
                raise ValueError(f"rating must be 1-5, got {self.rating}")

    @dataclass
    class ConfigResponse:
        profile: str
        weights: dict
        constraints: dict
        rl_config: dict

    @dataclass
    class ConfigUpdateRequest:
        profile: Optional[str] = None
        weights: Optional[dict] = None
        constraints: Optional[dict] = None

    @dataclass
    class MetricsResponse:
        total_requests: int
        total_feedback: int
        avg_score: float
        failure_count: int
        avg_latency_ms: float
        latency_p95_ms: float
        reward_breakdown: dict
