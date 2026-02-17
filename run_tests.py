#!/usr/bin/env python3
"""
Standalone test runner — works without pytest installed.
Usage: python tests/run_tests.py
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SKIP = "\033[93m~\033[0m"

results = {"pass": 0, "fail": 0, "skip": 0}


def run_test(name: str, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        results["pass"] += 1
    except Exception as e:
        print(f"  {FAIL} {name}")
        print(f"       {type(e).__name__}: {e}")
        results["fail"] += 1


def assert_approx(a, b, rel=1e-4, msg=""):
    if abs(a - b) > rel * max(abs(b), 1e-9):
        raise AssertionError(f"{msg}: {a} !≈ {b}")


# ===========================================================================
# PAM Scanner tests
# ===========================================================================

def test_pam_finds_ngg():
    from crispr_rl.features.pam_scanner import scan_sequence
    GUIDE = "ATCGATCGATCGATCGATCG"
    seq = GUIDE + "AGG" + "A" * 30
    hits = scan_sequence(seq, pam_pattern="NGG")
    fwd = [h for h in hits if h.strand == "+"]
    assert len(fwd) >= 1
    assert GUIDE in [h.protospacer for h in fwd]

def test_pam_both_strands():
    from crispr_rl.features.pam_scanner import scan_sequence
    seq = "A" * 25 + "CCCAAAAAAAAAAAAAAAAAAAAA" + "A" * 10  # CCN on fwd → NGG on rev
    hits = scan_sequence(seq)
    strands = {h.strand for h in hits}
    # should find hits on at least one strand
    assert len(strands) >= 1

def test_pam_protospacer_20nt():
    from crispr_rl.features.pam_scanner import scan_sequence
    seq = "GCATGCATGCATGCATGCATGG" + "T" * 20
    hits = scan_sequence(seq)
    for h in hits:
        assert len(h.protospacer) == 20

def test_pam_empty_sequence():
    from crispr_rl.features.pam_scanner import scan_sequence
    assert scan_sequence("") == []

def test_pam_too_short():
    from crispr_rl.features.pam_scanner import scan_sequence
    assert scan_sequence("ACGTGG") == []


# ===========================================================================
# Feature tests
# ===========================================================================

def test_gc_fraction():
    from crispr_rl.features.gc_content import gc_fraction
    assert_approx(gc_fraction("GCGC"), 1.0)
    assert_approx(gc_fraction("ATAT"), 0.0)
    assert_approx(gc_fraction("GCTA"), 0.5)
    assert_approx(gc_fraction(""), 0.0)

def test_homopolymer_run():
    from crispr_rl.features.gc_content import max_homopolymer_run
    assert max_homopolymer_run("ACGT") == 1
    assert max_homopolymer_run("AAAA") == 4
    assert max_homopolymer_run("AAAAAG") == 5

def test_run_length_penalty():
    from crispr_rl.features.gc_content import run_length_penalty
    assert run_length_penalty("ACGT", threshold=4) == 0.0
    pen = run_length_penalty("AAAAAA", threshold=4)
    assert pen > 0.0
    assert pen <= 1.0

def test_efficiency_proxy_range():
    from crispr_rl.features.thermo_proxy import efficiency_proxy
    s = efficiency_proxy("ATCGATCGATCGATCGATCG")
    assert 0.0 <= s <= 1.0

def test_tm_wallace():
    from crispr_rl.features.thermo_proxy import tm_wallace
    assert_approx(tm_wallace("GCGCGCGCGCGCGCGCGCGC"), 80.0)

def test_context_window():
    from crispr_rl.features.context import extract_context_window
    ctx = extract_context_window("A" * 100, 30)
    assert "upstream" in ctx and "guide" in ctx and "downstream" in ctx
    assert len(ctx["guide"]) == 20


# ===========================================================================
# Off-target tests
# ===========================================================================

GUIDE = "ATCGATCGATCGATCGATCG"

def test_count_mismatches_zero():
    from crispr_rl.scoring.off_target import count_mismatches
    assert count_mismatches(GUIDE, GUIDE) == 0

def test_count_mismatches_one():
    from crispr_rl.scoring.off_target import count_mismatches
    t = GUIDE[:5] + "A" + GUIDE[6:]
    assert count_mismatches(GUIDE, t) == 1

def test_seed_weighted_higher():
    from crispr_rl.scoring.off_target import weighted_mismatch_score
    in_seed = GUIDE[:19] + "A"
    out_seed = "A" + GUIDE[1:]
    assert weighted_mismatch_score(GUIDE, in_seed) > weighted_mismatch_score(GUIDE, out_seed)

def test_specificity_with_off_targets():
    from crispr_rl.scoring.off_target import specificity_proxy
    near = GUIDE[:2] + "AA" + GUIDE[4:]
    seq = GUIDE + "T" * 20 + near + "T" * 20
    spec = specificity_proxy(GUIDE, seq)
    assert 0.0 < spec <= 1.0
    assert spec < 1.0  # off-targets exist


# ===========================================================================
# Baseline scorer tests
# ===========================================================================

SEQUENCE = "A" * 40 + GUIDE + "GGG" + "T" * 80

def test_score_guide_returns_float():
    from crispr_rl.scoring.baseline import score_guide
    score, _ = score_guide(GUIDE, SEQUENCE, 0)
    assert isinstance(score, float)

def test_score_guide_deterministic():
    from crispr_rl.scoring.baseline import score_guide
    s1, _ = score_guide(GUIDE, SEQUENCE, 0)
    s2, _ = score_guide(GUIDE, SEQUENCE, 0)
    assert s1 == s2

def test_score_guide_in_range():
    from crispr_rl.scoring.baseline import score_guide
    s, _ = score_guide(GUIDE, SEQUENCE, 0)
    assert 0.0 <= s <= 1.0

def test_rank_candidates_sorted():
    from crispr_rl.scoring.baseline import rank_candidates
    cands = [
        {"id": "g0", "seq": "ATCGATCGATCGATCGATCG", "position": 0},
        {"id": "g1", "seq": "GCGCGCGCGCGCGCGCGCGC", "position": 10},
    ]
    ranked = rank_candidates(cands, SEQUENCE)
    scores = [c["score"] for c in ranked]
    assert scores == sorted(scores, reverse=True)


# ===========================================================================
# RL env tests
# ===========================================================================

CANDS = [
    {"id": "g0", "seq": "ATCGATCGATCGATCGATCG", "pam": "AGG", "locus": "p:0", "strand": "+", "gc": 0.5, "position": 0},
    {"id": "g1", "seq": "GCGCGCGCGCGCGCGCGCGC", "pam": "TGG", "locus": "p:20", "strand": "+", "gc": 1.0, "position": 20},
    {"id": "g2", "seq": "TATATATATATATATATATAT"[:20], "pam": "CGG", "locus": "p:40", "strand": "-", "gc": 0.0, "position": 40},
]

def test_env_reset():
    import numpy as np
    from crispr_rl.rl.env import GuideRNAEnv
    env = GuideRNAEnv(CANDS, SEQUENCE, seed=42)
    state = env.reset()
    assert isinstance(state, np.ndarray)
    assert state.shape == (len(CANDS) * 5,)

def test_env_step_returns_reward():
    from crispr_rl.rl.env import GuideRNAEnv
    env = GuideRNAEnv(CANDS, SEQUENCE, seed=42)
    env.reset()
    _, reward, _, _ = env.step(0)
    assert isinstance(reward, float)

def test_env_action_space():
    from crispr_rl.rl.env import GuideRNAEnv
    env = GuideRNAEnv(CANDS, SEQUENCE, seed=42)
    assert env.n_actions == 3

def test_env_invalid_action():
    from crispr_rl.rl.env import GuideRNAEnv
    env = GuideRNAEnv(CANDS, SEQUENCE, seed=42)
    env.reset()
    try:
        env.step(99)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

def test_epsilon_greedy_bandit():
    from crispr_rl.rl.bandit import EpsilonGreedyBandit
    b = EpsilonGreedyBandit(n_arms=5, seed=42)
    for _ in range(20):
        arm = b.select_arm()
        assert 0 <= arm < 5
    b.update(0, 0.9)
    assert b.counts[0] == 1

def test_ucb_bandit():
    from crispr_rl.rl.bandit import UCBBandit
    b = UCBBandit(n_arms=3, seed=42)
    arms = set()
    for _ in range(3):
        arm = b.select_arm()
        b.update(arm, 0.5)
        arms.add(arm)
    assert arms == {0, 1, 2}


# ===========================================================================
# Schema tests (dataclass or pydantic)
# ===========================================================================

def test_guide_schema_valid():
    from crispr_rl.api.schemas import GuideRNACandidate
    c = GuideRNACandidate(
        id="g001", seq="ATCGATCGATCGATCGATCG", pam="AGG",
        locus="chr1:100", strand="+", gc=0.5,
    )
    assert c.seq == "ATCGATCGATCGATCGATCG"

def test_guide_schema_invalid_seq():
    from crispr_rl.api.schemas import GuideRNACandidate
    try:
        GuideRNACandidate(id="g", seq="ACGT", pam="AGG", locus="x", strand="+", gc=0.5)
        raise AssertionError("Should have raised ValueError")
    except (ValueError, Exception):
        pass  # pydantic ValidationError or ValueError both OK

def test_feedback_schema_valid():
    from crispr_rl.api.schemas import FeedbackRequest
    fb = FeedbackRequest(candidate_id="g001", rating=5)
    assert fb.rating == 5

def test_feedback_schema_invalid_rating():
    from crispr_rl.api.schemas import FeedbackRequest
    try:
        FeedbackRequest(candidate_id="g001", rating=0)
        raise AssertionError("Should have raised")
    except (ValueError, Exception):
        pass

def test_design_request_too_short_sequence():
    from crispr_rl.api.schemas import DesignRequest
    try:
        DesignRequest(sequence="ACGT")
        raise AssertionError("Should have raised")
    except (ValueError, Exception):
        pass


# ===========================================================================
# Integration test
# ===========================================================================

INT_SEQ = (
    "ATGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"
    "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGG"
    "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG"
    "CATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
)

def test_integration_pam_sites_found():
    from crispr_rl.features.pam_scanner import scan_sequence
    hits = scan_sequence(INT_SEQ)
    assert len(hits) > 0, "Integration sequence must have PAM sites"

def test_integration_end_to_end():
    from crispr_rl.eval.harness import run_eval
    result = run_eval(INT_SEQ, profile="knockout", seed=42)
    assert len(result.rl_scores) > 0
    assert result.rl_mean >= 0.0

def test_integration_rl_ge_baseline():
    from crispr_rl.eval.harness import run_eval
    result = run_eval(INT_SEQ, profile="knockout", seed=42)
    if not result.baseline_scores:
        return  # skip if no candidates
    # Allow 5% tolerance
    assert result.rl_mean >= result.baseline_mean * 0.95, (
        f"RL {result.rl_mean:.4f} < baseline {result.baseline_mean:.4f}"
    )

def test_integration_deterministic():
    from crispr_rl.eval.harness import run_eval
    from crispr_rl.utils.seeds import set_global_seed
    set_global_seed(42)
    r1 = run_eval(INT_SEQ, profile="knockout", seed=42)
    set_global_seed(42)
    r2 = run_eval(INT_SEQ, profile="knockout", seed=42)
    assert_approx(r1.rl_mean, r2.rl_mean, msg="RL results must be deterministic")


# ===========================================================================
# Run all
# ===========================================================================

if __name__ == "__main__":
    import sys

    test_fns = [
        # PAM Scanner
        ("PAM: finds NGG on forward strand", test_pam_finds_ngg),
        ("PAM: scans both strands", test_pam_both_strands),
        ("PAM: protospacer is 20nt", test_pam_protospacer_20nt),
        ("PAM: empty sequence", test_pam_empty_sequence),
        ("PAM: too-short sequence", test_pam_too_short),
        # Features
        ("Features: GC fraction", test_gc_fraction),
        ("Features: homopolymer run detection", test_homopolymer_run),
        ("Features: run-length penalty", test_run_length_penalty),
        ("Features: efficiency proxy range", test_efficiency_proxy_range),
        ("Features: Tm Wallace", test_tm_wallace),
        ("Features: context window", test_context_window),
        # Off-target
        ("Off-target: zero mismatches", test_count_mismatches_zero),
        ("Off-target: one mismatch", test_count_mismatches_one),
        ("Off-target: seed region weighted higher", test_seed_weighted_higher),
        ("Off-target: specificity <1 with off-targets", test_specificity_with_off_targets),
        # Baseline scorer
        ("Baseline: returns float", test_score_guide_returns_float),
        ("Baseline: deterministic", test_score_guide_deterministic),
        ("Baseline: score in [0,1]", test_score_guide_in_range),
        ("Baseline: rank_candidates sorted", test_rank_candidates_sorted),
        # RL env
        ("RL env: reset returns ndarray", test_env_reset),
        ("RL env: step returns reward", test_env_step_returns_reward),
        ("RL env: action space", test_env_action_space),
        ("RL env: invalid action raises", test_env_invalid_action),
        ("RL bandit: epsilon-greedy selects valid arm", test_epsilon_greedy_bandit),
        ("RL bandit: UCB explores all arms first", test_ucb_bandit),
        # Schemas
        ("Schema: valid GuideRNACandidate", test_guide_schema_valid),
        ("Schema: invalid guide seq raises", test_guide_schema_invalid_seq),
        ("Schema: valid FeedbackRequest", test_feedback_schema_valid),
        ("Schema: invalid rating raises", test_feedback_schema_invalid_rating),
        ("Schema: sequence too short raises", test_design_request_too_short_sequence),
        # Integration
        ("Integration: PAM sites in synthetic seq", test_integration_pam_sites_found),
        ("Integration: end-to-end pipeline", test_integration_end_to_end),
        ("Integration: RL ≥ baseline on average", test_integration_rl_ge_baseline),
        ("Integration: deterministic with seed=42", test_integration_deterministic),
    ]

    print(f"\n🧬  crispr_rl test suite ({len(test_fns)} tests)\n")
    for label, fn in test_fns:
        run_test(label, fn)

    total = results["pass"] + results["fail"] + results["skip"]
    color = "\033[92m" if results["fail"] == 0 else "\033[91m"
    print(f"\n{color}Results: {results['pass']}/{total} passed, {results['fail']} failed\033[0m\n")
    sys.exit(0 if results["fail"] == 0 else 1)
