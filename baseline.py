"""Weighted-sum heuristic scorer v0 — baseline before RL optimization."""

from __future__ import annotations

from crispr_rl.features.gc_content import gc_fraction, gc_penalty, run_length_penalty
from crispr_rl.features.thermo_proxy import efficiency_proxy
from crispr_rl.features.context import positional_score
from crispr_rl.scoring.off_target import specificity_proxy

# Default weight vector
DEFAULT_W1 = 0.5   # efficiency weight
DEFAULT_W2 = 0.3   # off-target penalty weight
DEFAULT_W3 = 0.2   # coverage/positional weight


def score_guide(
    guide_seq: str,
    sequence: str,
    position: int,
    w1: float = DEFAULT_W1,
    w2: float = DEFAULT_W2,
    w3: float = DEFAULT_W3,
    gc_min: float = 0.4,
    gc_max: float = 0.6,
    max_homopolymer: int = 4,
) -> tuple[float, dict]:
    """
    Compute baseline score for a single guide.

    score = w1 * efficiency_proxy - w2 * off_target_penalty + w3 * coverage_proxy

    Returns (score, explanation_dict).
    """
    # Efficiency: thermodynamic proxy penalised by GC deviation and homopolymer runs
    thermo = efficiency_proxy(guide_seq)
    gc = gc_fraction(guide_seq)
    gc_pen = gc_penalty(gc, gc_min, gc_max)
    rl_pen = run_length_penalty(guide_seq, threshold=max_homopolymer)
    efficiency = max(0.0, thermo - gc_pen - rl_pen)

    # Off-target penalty (inverted specificity)
    specificity = specificity_proxy(guide_seq, sequence)
    off_target_penalty = 1.0 - specificity

    # Coverage proxy: positional score (prefer near TSS)
    coverage = positional_score(position, len(sequence))

    raw_score = w1 * efficiency - w2 * off_target_penalty + w3 * coverage
    # Normalise to [0, 1] approximately by clipping
    score = max(0.0, min(1.0, raw_score + w2))  # shift by max possible penalty

    explanation = {
        "efficiency": round(efficiency, 4),
        "thermo_tm_proxy": round(thermo, 4),
        "gc_fraction": round(gc, 4),
        "gc_penalty": round(gc_pen, 4),
        "homopolymer_penalty": round(rl_pen, 4),
        "specificity": round(specificity, 4),
        "off_target_penalty": round(off_target_penalty, 4),
        "coverage_positional": round(coverage, 4),
        "raw_score": round(raw_score, 4),
        "final_score": round(score, 4),
    }

    return score, explanation


def rank_candidates(
    candidates: list[dict],
    sequence: str,
    w1: float = DEFAULT_W1,
    w2: float = DEFAULT_W2,
    w3: float = DEFAULT_W3,
    max_candidates: int = 20,
) -> list[dict]:
    """
    Score and rank a list of candidate dicts (must have 'seq' and 'position' keys).

    Mutates each candidate in-place with 'score' and 'explanations' fields.
    Returns sorted list (descending score), truncated to *max_candidates*.
    """
    for cand in candidates:
        score, expl = score_guide(
            guide_seq=cand["seq"],
            sequence=sequence,
            position=cand.get("position", 0),
            w1=w1, w2=w2, w3=w3,
        )
        cand["score"] = score
        cand["explanations"] = expl

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:max_candidates]
