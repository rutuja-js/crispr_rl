"""Positional context extraction — window around guide, TSS distance proxy."""

from __future__ import annotations


WINDOW_SIZE = 30  # bp upstream and downstream for context


def extract_context_window(
    sequence: str,
    position: int,
    window: int = WINDOW_SIZE,
    guide_length: int = 20,
) -> dict[str, str]:
    """
    Extract flanking context around a guide RNA site.

    Returns dict with 'upstream', 'guide', 'downstream' subsequences.
    Sequences are padded with 'N' at boundaries.
    """
    seq = sequence.upper()
    n = len(seq)

    upstream_start = max(0, position - window)
    upstream_end = position
    guide_end = position + guide_length
    downstream_end = min(n, guide_end + window)

    upstream = seq[upstream_start:upstream_end].rjust(window, "N")
    guide = seq[position:guide_end] if guide_end <= n else seq[position:] + "N" * (guide_end - n)
    downstream = seq[guide_end:downstream_end].ljust(window, "N")

    return {
        "upstream": upstream[-window:],
        "guide": guide,
        "downstream": downstream[:window],
    }


def tss_distance_proxy(position: int, sequence_length: int) -> float:
    """
    Normalised distance from the 5' end of the sequence (TSS proxy).

    Returns a value in [0, 1] where 0 = near 5' end (TSS), 1 = near 3' end.
    In the absence of real annotation data, early positions in the gene body
    are treated as closer to the TSS.
    """
    if sequence_length <= 1:
        return 0.0
    return position / (sequence_length - 1)


def positional_score(
    position: int,
    sequence_length: int,
    preferred_frac: float = 0.2,
) -> float:
    """
    Return a positional score favouring sites near the 5' end of the coding sequence.

    *preferred_frac* = fraction of gene length considered 'near TSS'.
    Returns 1.0 for sites in the preferred window, decaying elsewhere.
    """
    frac = tss_distance_proxy(position, sequence_length)
    if frac <= preferred_frac:
        return 1.0
    return max(0.0, 1.0 - (frac - preferred_frac) / (1.0 - preferred_frac))
