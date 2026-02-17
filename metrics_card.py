"""Generate a metrics card comparing baseline vs RL across multiple seeds."""

from __future__ import annotations

import numpy as np

from crispr_rl.eval.harness import run_eval


def generate_metrics_card(
    sequence: str,
    profile: str = "knockout",
    seeds: list[int] | None = None,
) -> dict:
    """
    Run evaluation across multiple seeds and produce a summary metrics card.

    Returns dict with uplift %, latency stats, and score std-dev.
    """
    seeds = seeds or [42, 123, 777]
    results = [run_eval(sequence, profile=profile, seed=s) for s in seeds]

    uplifts = [r.uplift_pct for r in results]
    baseline_means = [r.baseline_mean for r in results]
    rl_means = [r.rl_mean for r in results]
    rl_latencies = [r.latency_ms_rl for r in results]
    baseline_latencies = [r.latency_ms_baseline for r in results]

    all_rl_scores = [s for r in results for s in r.rl_scores]
    all_baseline_scores = [s for r in results for s in r.baseline_scores]

    return {
        "profile": profile,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "uplift_pct_mean": round(float(np.mean(uplifts)), 2),
        "uplift_pct_std": round(float(np.std(uplifts)), 2),
        "baseline_score_mean": round(float(np.mean(baseline_means)), 4),
        "baseline_score_std": round(float(np.std(all_baseline_scores)), 4),
        "rl_score_mean": round(float(np.mean(rl_means)), 4),
        "rl_score_std": round(float(np.std(all_rl_scores)), 4),
        "latency_ms_rl_mean": round(float(np.mean(rl_latencies)), 2),
        "latency_ms_baseline_mean": round(float(np.mean(baseline_latencies)), 2),
        "per_seed": [
            {
                "seed": s,
                "baseline_mean": round(r.baseline_mean, 4),
                "rl_mean": round(r.rl_mean, 4),
                "uplift_pct": round(r.uplift_pct, 2),
                "latency_ms_rl": r.latency_ms_rl,
            }
            for s, r in zip(seeds, results)
        ],
    }
