"""Ablation study: toggle off-target, GC, context scoring components."""

from __future__ import annotations

from crispr_rl.eval.harness import run_eval, EvalResult


def run_ablations(sequence: str, profile: str = "knockout", seed: int = 42) -> dict[str, EvalResult]:
    """Run three ablation variants and return comparison dict."""
    results: dict[str, EvalResult] = {}

    # Full model
    results["full"] = run_eval(sequence, profile=profile, seed=seed)

    # No off-target (set w_specificity=0)
    from crispr_rl.utils.config import load_config, get_constraints, get_rl_config
    from crispr_rl.eval.harness import build_candidates_from_sequence
    from crispr_rl.scoring.baseline import rank_candidates
    from crispr_rl.scoring.composite import composite_reward, pareto_rerank
    from crispr_rl.rl.trainer import BanditTrainer
    import numpy as np
    import time

    cfg = load_config()
    constraints = get_constraints(cfg)
    rl_cfg = get_rl_config(cfg)

    for variant, weights in [
        ("no_off_target", {"w_efficiency": 0.7, "w_specificity": 0.0, "w_coverage": 0.3}),
        ("no_gc_context", {"w_efficiency": 0.8, "w_specificity": 0.2, "w_coverage": 0.0}),
    ]:
        cands = build_candidates_from_sequence(sequence, variant)
        if not cands:
            continue

        trainer = BanditTrainer(
            candidates=cands,
            sequence=sequence,
            weights=weights,
            constraints=constraints,
            policy="epsilon_greedy",
            rl_config=rl_cfg,
            seed=seed,
        )
        summary = trainer.train(n_steps=rl_cfg.get("n_training_steps", 500))
        rl_scores = [float(trainer.bandit.values[i]) for i in range(len(cands))]
        rl_mean = float(np.mean(rl_scores)) if rl_scores else 0.0
        results[variant] = EvalResult(
            baseline_scores=[],
            rl_scores=rl_scores,
            baseline_mean=0.0,
            rl_mean=rl_mean,
            uplift_pct=0.0,
            baseline_top5=[],
            rl_top5=cands[:5],
            latency_ms_baseline=0.0,
            latency_ms_rl=summary.get("latency_ms", 0.0),
        )

    return results
