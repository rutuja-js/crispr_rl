"""Baseline vs RL evaluation harness."""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np

from crispr_rl.data.loaders import load_gene_sequence
from crispr_rl.features.pam_scanner import scan_sequence
from crispr_rl.features.gc_content import gc_fraction, max_homopolymer_run
from crispr_rl.scoring.baseline import rank_candidates
from crispr_rl.scoring.composite import composite_reward, pareto_rerank
from crispr_rl.rl.trainer import BanditTrainer
from crispr_rl.utils.config import load_config, get_profile_weights, get_constraints, get_rl_config


class EvalResult(NamedTuple):
    baseline_scores: list[float]
    rl_scores: list[float]
    baseline_mean: float
    rl_mean: float
    uplift_pct: float
    baseline_top5: list[dict]
    rl_top5: list[dict]
    latency_ms_baseline: float
    latency_ms_rl: float


def build_candidates_from_sequence(sequence: str, request_id: str = "eval") -> list[dict]:
    hits = scan_sequence(sequence)
    candidates = []
    for i, hit in enumerate(hits):
        gc = gc_fraction(hit.protospacer)
        candidates.append({
            "id": f"guide_{request_id}_{i:03d}",
            "seq": hit.protospacer,
            "pam": hit.pam_seq,
            "locus": f"pos:{hit.position}",
            "strand": hit.strand,
            "gc": gc,
            "position": hit.position,
            "features": {"gc": gc, "position": hit.position},
            "risk_flags": [],
            "score": 0.0,
            "explanations": {},
        })
    return candidates


def run_eval(
    sequence: str,
    profile: str = "knockout",
    seed: int = 42,
    config_path: str | None = None,
) -> EvalResult:
    """Run baseline and RL evaluation on a sequence, return comparison metrics."""
    cfg = load_config(config_path)
    weights = get_profile_weights(cfg, profile)
    constraints = get_constraints(cfg)
    rl_cfg = get_rl_config(cfg)

    candidates_baseline = build_candidates_from_sequence(sequence, "baseline")
    candidates_rl = build_candidates_from_sequence(sequence, "rl")

    if not candidates_baseline:
        return EvalResult([], [], 0.0, 0.0, 0.0, [], [], 0.0, 0.0)

    # --- Baseline ---
    t0 = time.perf_counter()
    rank_candidates(
        candidates_baseline, sequence,
        w1=weights.get("w_efficiency", 0.5),
        w2=weights.get("w_specificity", 0.3),
        w3=weights.get("w_coverage", 0.2),
    )
    baseline_latency = (time.perf_counter() - t0) * 1000
    baseline_scores = [c["score"] for c in candidates_baseline[:20]]
    baseline_top5 = candidates_baseline[:5]

    # --- RL ---
    t0 = time.perf_counter()
    trainer = BanditTrainer(
        candidates=candidates_rl,
        sequence=sequence,
        weights=weights,
        constraints=constraints,
        policy="epsilon_greedy",
        rl_config=rl_cfg,
        seed=seed,
    )
    trainer.train(n_steps=rl_cfg.get("n_training_steps", 500))

    # Score all candidates with composite reward
    for i, cand in enumerate(candidates_rl):
        rl_reward = float(trainer.bandit.values[i])
        reward, components = composite_reward(
            cand["seq"], sequence, cand.get("position", 0), weights, constraints
        )
        cand["score"] = round(max(rl_reward, reward), 4)
        cand["reward"] = cand["score"]
        cand["components"] = components

    candidates_rl = pareto_rerank(candidates_rl, max_candidates=20)
    rl_latency = (time.perf_counter() - t0) * 1000
    rl_scores = [c["score"] for c in candidates_rl[:20]]
    rl_top5 = candidates_rl[:5]

    baseline_mean = float(np.mean(baseline_scores)) if baseline_scores else 0.0
    rl_mean = float(np.mean(rl_scores)) if rl_scores else 0.0
    uplift_pct = ((rl_mean - baseline_mean) / max(baseline_mean, 1e-9)) * 100

    return EvalResult(
        baseline_scores=baseline_scores,
        rl_scores=rl_scores,
        baseline_mean=baseline_mean,
        rl_mean=rl_mean,
        uplift_pct=uplift_pct,
        baseline_top5=baseline_top5,
        rl_top5=rl_top5,
        latency_ms_baseline=round(baseline_latency, 2),
        latency_ms_rl=round(rl_latency, 2),
    )
