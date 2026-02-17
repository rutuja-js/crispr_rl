"""PAM site detection with support for arbitrary regex PAM patterns."""

from __future__ import annotations

import re
from typing import NamedTuple

from crispr_rl.data.loaders import reverse_complement


class PAMHit(NamedTuple):
    position: int      # 0-based position of protospacer start on forward strand
    strand: str        # "+" or "-"
    protospacer: str   # 20nt guide sequence (5'→3' relative to guide)
    pam_seq: str       # actual PAM nucleotides found


# IUPAC codes used in PAM patterns
_IUPAC: dict[str, str] = {
    "N": "[ACGT]",
    "R": "[AG]",
    "Y": "[CT]",
    "S": "[GC]",
    "W": "[AT]",
    "K": "[GT]",
    "M": "[AC]",
    "B": "[CGT]",
    "D": "[AGT]",
    "H": "[ACT]",
    "V": "[ACG]",
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
}

GUIDE_LENGTH = 20


def _pam_to_regex(pam_pattern: str) -> re.Pattern:
    """Convert an IUPAC PAM string to a compiled regex pattern."""
    regex_str = "".join(_IUPAC.get(c.upper(), c) for c in pam_pattern)
    return re.compile(regex_str)


def scan_sequence(
    sequence: str,
    pam_pattern: str = "NGG",
    guide_length: int = GUIDE_LENGTH,
) -> list[PAMHit]:
    """
    Scan both strands of *sequence* for PAM sites and extract protospacers.

    For SpCas9 (NGG): protospacer is 20nt UPSTREAM of PAM on the strand carrying PAM.
    The function scans the forward strand and the reverse complement independently.

    Returns a list of PAMHit namedtuples sorted by position.
    """
    seq = sequence.upper()
    hits: list[PAMHit] = []
    pam_re = _pam_to_regex(pam_pattern)
    pam_len = len(pam_pattern)

    # --- Forward strand ---
    for m in pam_re.finditer(seq):
        pam_start = m.start()
        proto_start = pam_start - guide_length
        if proto_start < 0:
            continue  # not enough upstream sequence
        protospacer = seq[proto_start:pam_start]
        if len(protospacer) != guide_length:
            continue
        hits.append(PAMHit(
            position=proto_start,
            strand="+",
            protospacer=protospacer,
            pam_seq=m.group(),
        ))

    # --- Reverse strand (scan RC of sequence) ---
    rc_seq = reverse_complement(seq)
    seq_len = len(seq)
    for m in pam_re.finditer(rc_seq):
        pam_start = m.start()
        proto_start = pam_start - guide_length
        if proto_start < 0:
            continue
        protospacer = rc_seq[proto_start:pam_start]
        if len(protospacer) != guide_length:
            continue
        # Convert RC position back to forward-strand coordinate
        fwd_position = seq_len - (pam_start + pam_len)
        hits.append(PAMHit(
            position=fwd_position,
            strand="-",
            protospacer=protospacer,
            pam_seq=m.group(),
        ))

    hits.sort(key=lambda h: h.position)
    return hits
