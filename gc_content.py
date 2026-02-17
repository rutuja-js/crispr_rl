"""GC content calculation and homopolymer run-length penalties."""

from __future__ import annotations


def gc_fraction(seq: str) -> float:
    """Return fraction of G + C in the sequence. Returns 0.0 for empty string."""
    seq = seq.upper()
    if not seq:
        return 0.0
    gc = sum(1 for b in seq if b in "GC")
    return gc / len(seq)


def is_gc_optimal(gc: float, gc_min: float = 0.4, gc_max: float = 0.6) -> bool:
    """Return True if GC content is within the optimal range."""
    return gc_min <= gc <= gc_max


def max_homopolymer_run(seq: str) -> int:
    """Return the length of the longest homopolymer run in *seq*."""
    if not seq:
        return 0
    seq = seq.upper()
    max_run = 1
    current_run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def run_length_penalty(seq: str, threshold: int = 4) -> float:
    """
    Penalise homopolymer runs longer than *threshold*.

    Returns a penalty in [0, 1] where 0 = no penalty and 1 = maximum penalty.
    The penalty grows linearly with run length beyond the threshold.
    """
    max_run = max_homopolymer_run(seq)
    excess = max(0, max_run - threshold)
    # Normalise: 4 extra bases above threshold → penalty ≈ 0.8
    return min(1.0, excess * 0.2)


def gc_penalty(gc: float, gc_min: float = 0.4, gc_max: float = 0.6) -> float:
    """
    Return a continuous penalty for GC content outside the optimal window.

    Returns 0.0 if within bounds, linearly increasing outside.
    """
    if gc < gc_min:
        return gc_min - gc
    if gc > gc_max:
        return gc - gc_max
    return 0.0
