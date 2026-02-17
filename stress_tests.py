"""Stress tests: large FASTA, repetitive sequences, edge cases."""

from __future__ import annotations

import time

from crispr_rl.features.pam_scanner import scan_sequence
from crispr_rl.eval.harness import run_eval


def stress_large_sequence(length: int = 5000, seed: int = 42) -> dict:
    """Generate a random sequence of *length* bp and measure throughput."""
    import numpy as np
    rng = np.random.default_rng(seed)
    seq = "".join(rng.choice(list("ACGT"), size=length))

    t0 = time.perf_counter()
    hits = scan_sequence(seq)
    elapsed = (time.perf_counter() - t0) * 1000

    return {
        "sequence_length": length,
        "n_hits": len(hits),
        "scan_latency_ms": round(elapsed, 2),
    }


def stress_repetitive_sequence() -> dict:
    """Test with highly repetitive sequence (worst-case for off-target search)."""
    # 100× repeat of AAAAAAAAAAAAAAAAAAAANGG
    seq = ("AAAAAAAAAAAAAAAAAAAANGG" * 100).replace("N", "A")
    t0 = time.perf_counter()
    hits = scan_sequence(seq)
    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "sequence_length": len(seq),
        "n_hits": len(hits),
        "scan_latency_ms": round(elapsed, 2),
        "note": "repetitive polyA sequence",
    }


def stress_edge_cases() -> dict:
    """Run scanner on edge-case inputs and collect results."""
    results = {}

    # Empty-ish sequences
    for name, seq in [
        ("empty", ""),
        ("too_short", "ACGTGG"),
        ("exactly_23", "ACGTACGTACGTACGTACGTGG"),  # 20 nt + NGG
        ("no_pam", "A" * 100),
    ]:
        try:
            hits = scan_sequence(seq) if seq else []
            results[name] = {"n_hits": len(hits), "error": None}
        except Exception as e:
            results[name] = {"n_hits": 0, "error": str(e)}

    return results
