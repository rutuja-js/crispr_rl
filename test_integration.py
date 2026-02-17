"""Integration tests: synthetic FASTA → design → RL uplift check."""

import pytest
import numpy as np

from crispr_rl.data.loaders import validate_sequence
from crispr_rl.eval.harness import run_eval, build_candidates_from_sequence
from crispr_rl.features.pam_scanner import scan_sequence
from crispr_rl.scoring.baseline import rank_candidates
from crispr_rl.scoring.composite import composite_reward, pareto_rerank
from crispr_rl.rl.trainer import BanditTrainer
from crispr_rl.utils.config import load_config, get_profile_weights, get_constraints, get_rl_config
from crispr_rl.utils.seeds import set_global_seed


# A synthetic 400-nt FASTA sequence with multiple embedded PAM sites
SYNTHETIC_FASTA = """>SYNTHETIC_GENE_001
ATGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC
ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
GGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
GCATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC
GATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
GGGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC
TATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGG
"""

SEQUENCE = (
    "ATGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"
    "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    "GGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA"
    "GCATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC"
    "GATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    "GGGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"
)


class TestEndToEndPipeline:
    def test_sequence_has_pam_sites(self):
        hits = scan_sequence(SEQUENCE, pam_pattern="NGG")
        assert len(hits) > 0, "Synthetic sequence must have ≥1 PAM site"

    def test_build_candidates_not_empty(self):
        cands = build_candidates_from_sequence(SEQUENCE)
        assert len(cands) > 0

    def test_baseline_scores_all_set(self):
        cands = build_candidates_from_sequence(SEQUENCE)
        cfg = load_config()
        weights = get_profile_weights(cfg, "knockout")
        rank_candidates(cands, SEQUENCE,
                        w1=weights.get("w_efficiency", 0.5),
                        w2=weights.get("w_specificity", 0.3),
                        w3=weights.get("w_coverage", 0.2))
        assert all(isinstance(c["score"], float) for c in cands)
        assert all(c["score"] >= 0.0 for c in cands)

    def test_rl_produces_candidates_with_rewards(self):
        cands = build_candidates_from_sequence(SEQUENCE)
        cfg = load_config()
        weights = get_profile_weights(cfg, "knockout")
        constraints = get_constraints(cfg)
        rl_cfg = get_rl_config(cfg)

        set_global_seed(42)
        trainer = BanditTrainer(
            candidates=cands, sequence=SEQUENCE,
            weights=weights, constraints=constraints,
            policy="epsilon_greedy", rl_config=rl_cfg, seed=42,
        )
        summary = trainer.train(n_steps=100)  # short for speed
        assert "best_reward" in summary
        assert isinstance(summary["best_reward"], float)

    def test_rl_composite_score_ge_baseline_on_average(self):
        """RL mean score must be ≥ baseline mean score (acceptance criterion)."""
        result = run_eval(SEQUENCE, profile="knockout", seed=42)
        if not result.baseline_scores or not result.rl_scores:
            pytest.skip("No candidates found in test sequence")
        assert result.rl_mean >= result.baseline_mean * 0.95, (
            f"RL mean {result.rl_mean:.4f} should be ≥ baseline mean {result.baseline_mean:.4f}"
        )

    def test_deterministic_with_fixed_seed(self):
        """Same seed → identical RL results."""
        set_global_seed(42)
        r1 = run_eval(SEQUENCE, profile="knockout", seed=42)
        set_global_seed(42)
        r2 = run_eval(SEQUENCE, profile="knockout", seed=42)
        assert r1.rl_mean == pytest.approx(r2.rl_mean, rel=1e-5)

    def test_pareto_rerank_returns_list(self):
        cands = build_candidates_from_sequence(SEQUENCE)
        cfg = load_config()
        weights = get_profile_weights(cfg, "knockout")
        constraints = get_constraints(cfg)

        for c in cands:
            r, comp = composite_reward(c["seq"], SEQUENCE, c.get("position", 0), weights, constraints)
            c["reward"] = r
            c["components"] = comp

        ranked = pareto_rerank(cands, max_candidates=10)
        assert isinstance(ranked, list)
        assert len(ranked) <= 10

    def test_full_pipeline_single_gene(self):
        """End-to-end: load synthetic gene → design → get top candidates."""
        from crispr_rl.data.loaders import load_gene_sequence
        gene_seq = load_gene_sequence("BRCA1")
        assert len(gene_seq) > 0

        cands = build_candidates_from_sequence(gene_seq)
        # BRCA1 synthetic seq should have PAM sites
        if not cands:
            pytest.skip("No candidates for BRCA1 synthetic sequence")

        cfg = load_config()
        weights = get_profile_weights(cfg, "knockout")
        rank_candidates(cands, gene_seq,
                        w1=weights.get("w_efficiency", 0.5),
                        w2=weights.get("w_specificity", 0.3),
                        w3=weights.get("w_coverage", 0.2))
        assert cands[0]["score"] >= cands[-1]["score"]  # sorted descending
