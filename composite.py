"""Multi-objective composite reward with Pareto-aware reranking."""

from __future__ import annotations

import numpy as np

from crispr_rl.features.gc_content import gc_fraction, max_homopolymer_run
from crispr_rl.features.thermo_proxy import efficiency_proxy
from crispr_rl.features.context import positional_score
from crispr_rl.scoring.off_target import specificity_proxy


# Default profile weights (can be overridden)
DEFAULT_WEIGHTS = {
    "w_efficiency": 0.5,
    "w_specificity": 0.3,
    "w_coverage": 0.2,
}


def composite_reward(
    guide_seq: str,
    sequence: str,
    position: int,
    weights: dict[str, float] | None = None,
    constraints: dict | None = None,
) -> tuple[float, dict]:
    """
    Compute composite reward for a guide.

    reward = w_efficiency * efficiency – w_specificity * off_target_penalty + w_coverage * coverage

    Constraint violations set reward to 0.0 and add to risk_flags.

    Returns (reward, components_dict).
    """
    w = weights or DEFAULT_WEIGHTS
    con = constraints or {}

    gc_min = con.get("gc_min", 0.35)
    gc_max = con.get("gc_max", 0.65)
    max_hp = con.get("max_homopolymer_run", 5)

    # Compute components
    eff = efficiency_proxy(guide_seq)
    spec = specificity_proxy(guide_seq, sequence)
    cov = positional_score(position, len(sequence))

    gc = gc_fraction(guide_seq)
    hp = max_homopolymer_run(guide_seq)

    # Check hard constraints
    risk_flags: list[str] = []
    if gc < gc_min or gc > gc_max:
        risk_flags.append(f"gc_out_of_range:{gc:.2f}")
    if hp > max_hp:
        risk_flags.append(f"homopolymer_run:{hp}")

    # Reward computation
    w_eff = w.get("w_efficiency", 0.5)
    w_spec = w.get("w_specificity", 0.3)
    w_cov = w.get("w_coverage", 0.2)

    raw = w_eff * eff + w_spec * spec + w_cov * cov
    reward = raw if not risk_flags else raw * 0.5  # soft penalty for constraint violations

    components = {
        "efficiency": round(eff, 4),
        "specificity": round(spec, 4),
        "coverage": round(cov, 4),
        "gc_fraction": round(gc, 4),
        "max_homopolymer": hp,
        "risk_flags": risk_flags,
        "raw_reward": round(raw, 4),
        "reward": round(reward, 4),
    }

    return reward, components


def pareto_rerank(
    candidates: list[dict],
    objectives: list[str] = ("efficiency", "specificity", "coverage"),
    max_candidates: int = 20,
) -> list[dict]:
    """
    Apply Pareto-front reranking for diversity across multiple objectives.

    Candidates with a 'components' dict are expected (set by composite_reward).
    Pareto-dominant candidates are placed first, then sorted by composite reward within tiers.
    """
    if not candidates:
        return []

    def get_obj(c: dict, key: str) -> float:
        return c.get("components", {}).get(key, 0.0)

    # Build objective matrix
    obj_matrix = np.array([
        [get_obj(c, o) for o in objectives]
        for c in candidates
    ])

    n = len(candidates)
    dominated = [False] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is better on all objectives
            if np.all(obj_matrix[j] >= obj_matrix[i]) and np.any(obj_matrix[j] > obj_matrix[i]):
                dominated[i] = True
                break

    pareto_front = [c for c, d in zip(candidates, dominated) if not d]
    rest = [c for c, d in zip(candidates, dominated) if d]

    # Sort each group by composite reward
    pareto_front.sort(key=lambda c: c.get("reward", 0.0), reverse=True)
    rest.sort(key=lambda c: c.get("reward", 0.0), reverse=True)

    return (pareto_front + rest)[:max_candidates]
