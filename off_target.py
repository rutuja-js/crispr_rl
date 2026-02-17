"""Off-target mismatch search with seed-region weighting."""

from __future__ import annotations

from crispr_rl.data.loaders import reverse_complement

SEED_REGION_LENGTH = 12   # last N bases before PAM — most critical for specificity
SEED_WEIGHT = 3.0         # seed mismatches count N× heavier
SPECIFICITY_THRESHOLD = 0.3  # prune guides below this value


def count_mismatches(guide: str, target: str) -> int:
    """Count position-wise mismatches between two equal-length strings."""
    if len(guide) != len(target):
        raise ValueError("Guide and target must have equal length for mismatch counting.")
    return sum(1 for a, b in zip(guide.upper(), target.upper()) if a != b)


def weighted_mismatch_score(
    guide: str,
    target: str,
    seed_region_length: int = SEED_REGION_LENGTH,
    seed_weight: float = SEED_WEIGHT,
) -> float:
    """
    Compute a weighted mismatch score where seed-region mismatches are heavier.

    Returns a float where 0 = perfect match (worst for off-target), higher = more mismatches.
    """
    guide = guide.upper()
    target = target.upper()
    n = len(guide)
    seed_start = n - seed_region_length  # seed = last `seed_region_length` bases

    score = 0.0
    for i, (g, t) in enumerate(zip(guide, target)):
        if g != t:
            weight = seed_weight if i >= seed_start else 1.0
            score += weight
    return score


def find_off_targets(
    guide: str,
    sequence: str,
    max_mismatches: int = 3,
    seed_region_length: int = SEED_REGION_LENGTH,
    seed_weight: float = SEED_WEIGHT,
) -> list[dict]:
    """
    Search *sequence* (and its RC) for near-matches to *guide*.

    Returns list of dicts: {position, strand, target_seq, mismatch_count, weighted_score}.
    A perfect match (mismatch_count==0) at the on-target site is expected and included.
    """
    guide = guide.upper()
    guide_len = len(guide)
    seq = sequence.upper()
    rc_seq = reverse_complement(seq)
    hits: list[dict] = []

    for strand, s in [("+", seq), ("-", rc_seq)]:
        for i in range(len(s) - guide_len + 1):
            window = s[i : i + guide_len]
            if len(window) < guide_len:
                continue
            mm = count_mismatches(guide, window)
            if mm <= max_mismatches:
                ws = weighted_mismatch_score(guide, window, seed_region_length, seed_weight)
                fwd_pos = i if strand == "+" else len(seq) - i - guide_len
                hits.append({
                    "position": fwd_pos,
                    "strand": strand,
                    "target_seq": window,
                    "mismatch_count": mm,
                    "weighted_score": ws,
                })

    return hits


def specificity_proxy(
    guide: str,
    sequence: str,
    max_mismatches: int = 3,
    seed_region_length: int = SEED_REGION_LENGTH,
    seed_weight: float = SEED_WEIGHT,
) -> float:
    """
    Return a specificity score in (0, 1].

    Defined as 1 / (1 + sum_of_weighted_off_target_scores).
    The on-target site (mismatch_count == 0) is excluded from the penalty sum.
    """
    hits = find_off_targets(guide, sequence, max_mismatches, seed_region_length, seed_weight)
    # Exclude the on-target match (mismatch_count == 0)
    off_target_score_sum = sum(h["weighted_score"] for h in hits if h["mismatch_count"] > 0)
    return 1.0 / (1.0 + off_target_score_sum)


def passes_specificity_filter(
    guide: str,
    sequence: str,
    threshold: float = SPECIFICITY_THRESHOLD,
) -> bool:
    """Return True if the guide's specificity_proxy is above *threshold*."""
    return specificity_proxy(guide, sequence) >= threshold
