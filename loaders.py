"""FASTA/gene ID loaders and sequence utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    """Return reverse complement of a DNA sequence."""
    return seq.translate(COMPLEMENT)[::-1]


def validate_sequence(seq: str) -> str:
    """Uppercase and validate that sequence only contains ACGTN."""
    seq = seq.upper().strip()
    invalid = set(seq) - set("ACGTN")
    if invalid:
        raise ValueError(f"Sequence contains invalid characters: {invalid}")
    return seq


def parse_fasta_string(fasta_text: str) -> dict[str, str]:
    """Parse a FASTA-formatted string into {header: sequence} dict."""
    records: dict[str, str] = {}
    current_header: str | None = None
    current_seq: list[str] = []

    for line in fasta_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                records[current_header] = "".join(current_seq).upper()
            current_header = line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line)

    if current_header is not None:
        records[current_header] = "".join(current_seq).upper()

    return records


def parse_fasta_file(path: str | Path) -> dict[str, str]:
    """Parse a FASTA file from disk."""
    with open(path) as f:
        return parse_fasta_string(f.read())


# ---------------------------------------------------------------------------
# Simulated gene loader (for demo/testing when no real FASTA available)
# ---------------------------------------------------------------------------

# Simple synthetic sequences keyed by common gene names for demo purposes
_SYNTHETIC_GENE_SEQS: dict[str, str] = {
    "BRCA1": (
        "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAG"
        "AGTGTCCCATCTGTCTGGAGTTGATCAAGGAACCTGTCTCCACAAAGTGTGACCACATATTTTGCAAA"
        "TTTTGCATGCCAAAAAATAAAATAGTGAGATGAATAGTTTAATTAGAGAGCAGATTTGAAACACTCTTT"
        "TTAAAGTTATGGGAAGATCTTGGAATTAATTTTTGTAAAGAAGACATGAATAAAGCCTTGAATAAACAC"
        "ATGTGTAAATAAGTTTTTAAATGTAAATGCATGCATTTAAATTTTTTTAATTTTTATAGTGAATTTTTTT"
    ),
    "TP53": (
        "ATGGAGGAGCCGCAGTCAGATCCTAGCATAGTGAGTCGTATTGAGTCCAAAAAGGAAATTTGCAAAGC"
        "CCTGCACCAGCAGCTCCACAGGTAAGAAATTTGCAAAGAATTTTTGCCTTGTGCCACTGGCCTTTCAG"
        "ATGGCTGGCAAAGGGAATGAGATTTTTTGAAAGCCCGGAGCTACTTCAGCATGATGATGGTGAGGATGG"
        "TCTTGGAGAGGCCAGGAGCCTGTATGAGCGCATCTTTATCTTCCTACAGACCGGCGCACAGAGGAAGAG"
        "AATCTCCGCAAGAAAGGGGAGCCTCACCACGAGCTGCCCCCAGGGAGCACTAAGCGAGCACTGTCCGTG"
    ),
    "EGFR": (
        "ATGCGACCCTCCGGGACGGCCGGGGCAGCGCTCCTGGCGCTGCTGGCTGCGCTCTGCCCGGCGAGTCG"
        "GGCTCTGGAGGAAAAGAAAGTTTGCCAAGGCACGAGTAACAAGCTCACGCAGTTGGGCACTTTTGAAGA"
        "CAAATTTATCTGTTGTGAAGGAAATCAAAGAGCTGCAGCAGTTCAGCAACAACACCAATGTCTGCAGTT"
        "CACGGACAATGAGCTGGTGGAGAATGACATCACAGAGAAAGAGATCCTGCCTGTGGACATCAGCATGGC"
        "CATCAGCAAATGCAAGCAAGGGATCCTCAAGAGCTTGGAAGAGCAAGATGAAAATCCAGAGAAAGGGCC"
    ),
}


def load_gene_sequence(gene_id: str, organism: str = "human") -> str:
    """
    Load a gene sequence. Uses synthetic sequences for demo purposes.
    In production, would query Ensembl/NCBI or a local database.
    """
    key = gene_id.upper()
    if key in _SYNTHETIC_GENE_SEQS:
        return validate_sequence(_SYNTHETIC_GENE_SEQS[key])
    # Generate a deterministic pseudo-random sequence for unknown genes
    import hashlib
    h = int(hashlib.md5(f"{gene_id}_{organism}".encode()).hexdigest(), 16)
    bases = "ACGT"
    length = 400
    seq = "".join(bases[(h >> (2 * i)) & 3] for i in range(length))
    return seq
