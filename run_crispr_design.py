#!/usr/bin/env python3
"""
crispr_rl demo script

Usage:
    python demo/run_crispr_design.py --sequence ATCG...    --profile knockout
    python demo/run_crispr_design.py --gene_ids BRCA1 TP53 --profile knockout
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure package is on path when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crispr_rl.data.loaders import load_gene_sequence, validate_sequence
from crispr_rl.eval.harness import run_eval, build_candidates_from_sequence
from crispr_rl.eval.metrics_card import generate_metrics_card
from crispr_rl.scoring.baseline import rank_candidates
from crispr_rl.scoring.composite import composite_reward, pareto_rerank
from crispr_rl.rl.trainer import BanditTrainer
from crispr_rl.utils.config import load_config, get_profile_weights, get_constraints, get_rl_config
from crispr_rl.utils.logging import get_logger, new_run_id
from crispr_rl.utils.seeds import set_global_seed

logger = get_logger("crispr_rl.demo")


def print_candidates_table(candidates: list[dict], title: str, top_n: int = 5) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    header = f"{'#':>3}  {'ID':<24} {'Seq (first 10nt)':<14} {'GC':>5}  {'Score':>7}  {'Strand':>6}"
    print(header)
    print("-" * 70)
    for i, c in enumerate(candidates[:top_n]):
        seq_short = c.get("seq", "")[:10] + "…"
        gc = c.get("gc", 0.0)
        score = c.get("score", c.get("reward", 0.0))
        strand = c.get("strand", "?")
        cid = c.get("id", f"guide_{i:03d}")
        print(f"{i+1:>3}  {cid:<24} {seq_short:<14} {gc:>5.2f}  {score:>7.4f}  {strand:>6}")
    print()


def print_comparison_table(baseline_top5: list[dict], rl_top5: list[dict]) -> None:
    print(f"\n{'='*70}")
    print("  SIDE-BY-SIDE COMPARISON — Baseline vs RL (top 5)")
    print(f"{'='*70}")
    header = f"{'Rank':>4}  {'Baseline Score':>14}  {'RL Score':>10}  {'ΔScore':>8}"
    print(header)
    print("-" * 50)
    for i in range(min(5, len(baseline_top5), len(rl_top5))):
        b = baseline_top5[i].get("score", 0.0)
        r = rl_top5[i].get("score", rl_top5[i].get("reward", 0.0))
        delta = r - b
        sign = "+" if delta >= 0 else ""
        print(f"{i+1:>4}  {b:>14.4f}  {r:>10.4f}  {sign}{delta:>7.4f}")
    print()


def export_results(candidates: list[dict], output_dir: Path, prefix: str = "design_results") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / f"{prefix}.csv"
    if candidates:
        keys = ["id", "seq", "pam", "locus", "strand", "gc", "score"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(candidates)
        print(f"  ✓ CSV  → {csv_path}")

    # JSON
    json_path = output_dir / f"{prefix}.json"
    with open(json_path, "w") as f:
        json.dump(candidates, f, indent=2, default=str)
    print(f"  ✓ JSON → {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="crispr_rl demo")
    parser.add_argument("--sequence", type=str, default=None, help="Raw DNA sequence string")
    parser.add_argument("--gene_ids", nargs="+", default=None, help="Gene IDs (e.g. BRCA1 TP53)")
    parser.add_argument(
        "--profile",
        choices=["knockout", "knockdown", "screening"],
        default="knockout",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    set_global_seed(args.seed)
    run_id = new_run_id()
    print(f"\n🧬  crispr_rl Demo  |  run_id={run_id}  |  profile={args.profile}  |  seed={args.seed}")

    # ------------------------------------------------------------------
    # 1. Resolve sequence
    # ------------------------------------------------------------------
    if args.sequence:
        try:
            sequence = validate_sequence(args.sequence)
        except ValueError as e:
            print(f"\n❌ Invalid sequence: {e}")
            sys.exit(1)
        print(f"\n→ Using provided sequence ({len(sequence)} nt)")
    elif args.gene_ids:
        print(f"\n→ Loading gene sequences for: {args.gene_ids}")
        parts = [load_gene_sequence(g) for g in args.gene_ids]
        sequence = ("N" * 10).join(parts)
        print(f"   Combined sequence length: {len(sequence)} nt")
    else:
        # Use a synthetic default sequence with known PAM sites
        sequence = (
            "ATGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCATCGATCGATCGATCGATCGATCGATCG"
            "TAGCTAGCTAGCATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGGG"
            "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGGG"
            "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"
            "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGGG"
        )
        print(f"\n→ Using built-in demo sequence ({len(sequence)} nt)")

    # ------------------------------------------------------------------
    # 2. Load config
    # ------------------------------------------------------------------
    cfg = load_config()
    weights = get_profile_weights(cfg, args.profile)
    constraints = get_constraints(cfg)
    rl_cfg = get_rl_config(cfg)
    print(f"\n→ Config loaded | weights: {weights}")

    # ------------------------------------------------------------------
    # 3. Run baseline scorer
    # ------------------------------------------------------------------
    print("\n─── Step 1/3: Baseline Scorer ────────────────────────────────")
    baseline_candidates = build_candidates_from_sequence(sequence, f"baseline_{run_id}")
    if not baseline_candidates:
        print("❌ No candidates found. Sequence may be too short or lack PAM sites.")
        sys.exit(1)

    rank_candidates(
        baseline_candidates, sequence,
        w1=weights.get("w_efficiency", 0.5),
        w2=weights.get("w_specificity", 0.3),
        w3=weights.get("w_coverage", 0.2),
    )
    print_candidates_table(baseline_candidates, "Baseline Top-5 Candidates")

    # ------------------------------------------------------------------
    # 4. Run RL optimizer
    # ------------------------------------------------------------------
    print("─── Step 2/3: RL Optimizer (500 steps) ──────────────────────")
    t0 = time.perf_counter()
    rl_candidates = build_candidates_from_sequence(sequence, f"rl_{run_id}")

    trainer = BanditTrainer(
        candidates=rl_candidates,
        sequence=sequence,
        weights=weights,
        constraints=constraints,
        policy="epsilon_greedy",
        rl_config=rl_cfg,
        seed=args.seed,
        run_id=run_id,
    )
    summary = trainer.train(n_steps=rl_cfg.get("n_training_steps", 500))
    rl_elapsed = time.perf_counter() - t0

    # Annotate with composite rewards
    for i, cand in enumerate(rl_candidates):
        bv = float(trainer.bandit.values[i])
        reward, components = composite_reward(
            cand["seq"], sequence, cand.get("position", 0), weights, constraints
        )
        cand["score"] = round(max(bv, reward), 4)
        cand["reward"] = cand["score"]
        cand["components"] = components

    rl_candidates = pareto_rerank(rl_candidates, max_candidates=20)
    print_candidates_table(rl_candidates, "RL Optimized Top-5 Candidates")

    # ------------------------------------------------------------------
    # 5. Side-by-side comparison
    # ------------------------------------------------------------------
    print("─── Step 3/3: Comparison ─────────────────────────────────────")
    print_comparison_table(baseline_candidates[:5], rl_candidates[:5])

    baseline_mean = np.mean([c.get("score", 0.0) for c in baseline_candidates[:20]]) if baseline_candidates else 0
    rl_mean = np.mean([c.get("score", 0.0) for c in rl_candidates[:20]]) if rl_candidates else 0
    uplift_pct = ((rl_mean - baseline_mean) / max(baseline_mean, 1e-9)) * 100

    print(f"  Baseline mean score : {baseline_mean:.4f}")
    print(f"  RL mean score       : {rl_mean:.4f}")
    print(f"  Uplift              : {uplift_pct:+.1f}%")
    print(f"  RL latency          : {rl_elapsed*1000:.0f} ms")

    # ------------------------------------------------------------------
    # 6. Metrics card (3 seeds)
    # ------------------------------------------------------------------
    print("\n─── Metrics Card (3 seeds: 42, 123, 777) ─────────────────────")
    card = generate_metrics_card(sequence, profile=args.profile, seeds=[42, 123, 777])
    print(f"  Uplift mean  ± std : {card['uplift_pct_mean']:+.1f}% ± {card['uplift_pct_std']:.1f}%")
    print(f"  RL score mean± std : {card['rl_score_mean']:.4f} ± {card['rl_score_std']:.4f}")
    print(f"  RL latency (avg)   : {card['latency_ms_rl_mean']:.0f} ms")
    for row in card["per_seed"]:
        print(f"    seed={row['seed']}: baseline={row['baseline_mean']:.4f}  rl={row['rl_mean']:.4f}  uplift={row['uplift_pct']:+.1f}%")

    # ------------------------------------------------------------------
    # 7. Export
    # ------------------------------------------------------------------
    print(f"\n─── Exporting Results ────────────────────────────────────────")
    all_export = []
    for c in rl_candidates:
        all_export.append({
            "id": c.get("id", ""),
            "seq": c.get("seq", ""),
            "pam": c.get("pam", ""),
            "locus": c.get("locus", ""),
            "strand": c.get("strand", ""),
            "gc": c.get("gc", 0.0),
            "score": c.get("score", 0.0),
            "risk_flags": c.get("risk_flags", []),
            "explanations": c.get("explanations", {}),
        })
    export_results(all_export, Path(args.output_dir))
    print(f"\n✅  Demo complete — run_id={run_id}\n")


if __name__ == "__main__":
    main()
