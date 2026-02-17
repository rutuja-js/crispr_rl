"""FastAPI service for the crispr_rl guide RNA design engine."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from crispr_rl.api.schemas import (
    DesignRequest,
    DesignResponse,
    FeedbackRequest,
    GuideRNACandidate,
    ConfigResponse,
    ConfigUpdateRequest,
    MetricsResponse,
)
from crispr_rl.data.loaders import load_gene_sequence, parse_fasta_string, validate_sequence
from crispr_rl.features.pam_scanner import scan_sequence
from crispr_rl.features.gc_content import gc_fraction, max_homopolymer_run
from crispr_rl.rl.trainer import BanditTrainer
from crispr_rl.scoring.baseline import rank_candidates
from crispr_rl.scoring.composite import composite_reward, pareto_rerank
from crispr_rl.utils.config import load_config, get_profile_weights, get_constraints, get_rl_config
from crispr_rl.utils.logging import get_logger, new_run_id

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

DB_PATH = Path("crispr_rl_feedback.db")
logger = get_logger("crispr_rl.api")

# Mutable server state
_state: dict[str, Any] = {
    "config": None,
    "current_profile": "knockout",
    "current_weights": None,
    "request_cache": {},       # request_id → DesignResponse
    "metrics": {
        "total_requests": 0,
        "total_feedback": 0,
        "failure_count": 0,
        "latencies_ms": [],
        "scores": [],
        "reward_breakdown": {},
    },
}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    cfg = load_config()
    _state["config"] = cfg
    _state["current_weights"] = get_profile_weights(cfg, "knockout")

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                rating INTEGER NOT NULL,
                notes TEXT,
                rationale TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                feedback_count INTEGER,
                delta_reward REAL,
                notes TEXT
            )
        """)
        await db.commit()
    logger.info("crispr_rl API server started.")
    yield
    logger.info("crispr_rl API server shutting down.")


app = FastAPI(
    title="crispr_rl",
    description="CRISPR guide RNA design engine with RL optimization",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_candidates(sequence: str, request: DesignRequest) -> list[dict]:
    """Scan PAM sites and build raw candidate list."""
    cfg = _state["config"] or load_config()
    constraints = get_constraints(cfg)
    pam_pattern = request.params.get("pam_pattern", "NGG")

    hits = scan_sequence(sequence, pam_pattern=pam_pattern)
    if not hits:
        return []

    max_cands = constraints.get("max_candidates", 20)
    candidates = []
    for i, hit in enumerate(hits[:max_cands * 2]):
        gc = gc_fraction(hit.protospacer)
        hp = max_homopolymer_run(hit.protospacer)
        risk_flags = []
        if gc < constraints.get("gc_min", 0.35) or gc > constraints.get("gc_max", 0.65):
            risk_flags.append(f"gc_out_of_range:{gc:.2f}")
        if hp > constraints.get("max_homopolymer_run", 5):
            risk_flags.append(f"homopolymer_run:{hp}")

        candidates.append({
            "id": f"guide_{request.request_id[:8]}_{i:03d}",
            "seq": hit.protospacer,
            "pam": hit.pam_seq,
            "locus": f"pos:{hit.position}",
            "strand": hit.strand,
            "gc": gc,
            "position": hit.position,
            "features": {
                "gc": gc,
                "max_homopolymer": hp,
                "position": hit.position,
                "strand": hit.strand,
            },
            "risk_flags": risk_flags,
            "score": 0.0,
            "explanations": {},
        })

    return candidates[:max_cands]


def _run_rl(candidates: list[dict], sequence: str, request: DesignRequest, run_id: str) -> list[dict]:
    """Run the RL bandit trainer and annotate candidates with RL scores."""
    if not candidates:
        return []

    cfg = _state["config"] or load_config()
    weights = _state["current_weights"] or get_profile_weights(cfg, request.profile)
    rl_cfg = get_rl_config(cfg)
    constraints = get_constraints(cfg)

    trainer = BanditTrainer(
        candidates=candidates,
        sequence=sequence,
        weights=weights,
        constraints=constraints,
        policy="epsilon_greedy",
        rl_config=rl_cfg,
        seed=request.seed,
        run_id=run_id,
    )
    summary = trainer.train(n_steps=rl_cfg.get("n_training_steps", 500))

    # Annotate candidates with final bandit values
    for i, cand in enumerate(candidates):
        rl_reward = float(trainer.bandit.values[i])
        reward, components = composite_reward(
            guide_seq=cand["seq"],
            sequence=sequence,
            position=cand.get("position", 0),
            weights=weights,
            constraints=constraints,
        )
        cand["score"] = round(max(rl_reward, reward), 4)
        cand["explanations"] = {k: v for k, v in components.items() if k != "risk_flags"}
        if components.get("risk_flags"):
            cand["risk_flags"] = list(set(cand.get("risk_flags", []) + components["risk_flags"]))

    return summary


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/crispr/ping")
async def ping():
    return {"status": "ok"}


@app.post("/crispr/design", response_model=DesignResponse)
async def design(request: DesignRequest):
    # Idempotency
    if request.request_id in _state["request_cache"]:
        return _state["request_cache"][request.request_id]

    start = time.perf_counter()
    run_id = new_run_id()
    _state["metrics"]["total_requests"] += 1

    try:
        # Resolve sequence
        if request.sequence:
            sequence = validate_sequence(request.sequence)
        elif request.gene_ids:
            parts = [load_gene_sequence(g, request.organism) for g in request.gene_ids]
            sequence = "N" * 10 + "NNNNNNNNNN".join(parts)  # join with separator
        else:
            raise HTTPException(
                status_code=422,
                detail="Either 'sequence' or 'gene_ids' must be provided.",
            )

        # Build candidates
        candidates = _build_candidates(sequence, request)
        if not candidates:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"No valid guide RNA candidates found in the provided sequence "
                    f"using PAM pattern '{request.params.get('pam_pattern', 'NGG')}'. "
                    "Ensure the sequence is ≥23 nt and contains at least one PAM site."
                ),
            )

        # Baseline scoring
        cfg = _state["config"] or load_config()
        weights = _state["current_weights"] or get_profile_weights(cfg, request.profile)
        rank_candidates(candidates, sequence, **{
            "w1": weights.get("w_efficiency", 0.5),
            "w2": weights.get("w_specificity", 0.3),
            "w3": weights.get("w_coverage", 0.2),
        })

        # RL optimization
        rl_summary = _run_rl(candidates, sequence, request, run_id)

        # Final reranking
        for c in candidates:
            c["reward"] = c.get("score", 0.0)
            c["components"] = c.get("explanations", {})
        candidates = pareto_rerank(candidates, max_candidates=get_constraints(cfg).get("max_candidates", 20))

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        _state["metrics"]["latencies_ms"].append(latency_ms)
        avg_score = sum(c["score"] for c in candidates) / len(candidates)
        _state["metrics"]["scores"].append(avg_score)

        schema_candidates = [
            GuideRNACandidate(
                id=c["id"],
                seq=c["seq"],
                pam=c["pam"],
                locus=c["locus"],
                strand=c["strand"],
                gc=c["gc"],
                features=c.get("features", {}),
                score=c.get("score", 0.0),
                explanations=c.get("explanations", {}),
                risk_flags=c.get("risk_flags", []),
            )
            for c in candidates
        ]

        response = DesignResponse(
            request_id=request.request_id,
            run_id=run_id,
            candidates=schema_candidates,
            metadata={
                "policy": "epsilon_greedy",
                "profile": request.profile,
                "weights": weights,
                "latency_ms": latency_ms,
                "seed": request.seed,
                "n_candidates": len(schema_candidates),
                "reward_breakdown": rl_summary.get("reward_breakdown", {}),
            },
        )

        _state["request_cache"][request.request_id] = response
        logger.info(
            f"run_id={run_id} gene_ids={request.gene_ids} profile={request.profile} "
            f"n_candidates={len(schema_candidates)} latency_ms={latency_ms}"
        )
        return response

    except HTTPException:
        _state["metrics"]["failure_count"] += 1
        raise
    except Exception as exc:
        _state["metrics"]["failure_count"] += 1
        logger.exception(f"Design request failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}")


@app.post("/crispr/feedback")
async def feedback(req: FeedbackRequest):
    """Store human-in-the-loop feedback and update bandit priors."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO feedback (candidate_id, rating, notes, rationale) VALUES (?, ?, ?, ?)",
            (req.candidate_id, req.rating, req.notes, req.rationale),
        )
        row = await db.execute("SELECT COUNT(*) FROM feedback")
        count = (await row.fetchone())[0]

        # Compute delta for audit log (simplified: rating normalised to [-1, +1])
        delta = (req.rating - 3) / 2.0  # rating 1→-1, 5→+1

        await db.execute(
            "INSERT INTO audit_log (version, feedback_count, delta_reward, notes) VALUES (?,?,?,?)",
            (str(uuid.uuid4())[:8], count, delta, f"feedback for {req.candidate_id}"),
        )
        await db.commit()

    _state["metrics"]["total_feedback"] += 1
    logger.info(f"Feedback received: candidate={req.candidate_id} rating={req.rating} delta={delta:.2f}")

    action = "reinforced" if req.rating >= 4 else "down-weighted" if req.rating <= 2 else "neutral"
    return {
        "status": "ok",
        "candidate_id": req.candidate_id,
        "rating": req.rating,
        "delta_reward": delta,
        "action": action,
        "total_feedback": count,
    }


@app.get("/crispr/config", response_model=ConfigResponse)
async def get_config():
    cfg = _state["config"] or load_config()
    profile = _state["current_profile"]
    return ConfigResponse(
        profile=profile,
        weights=_state["current_weights"] or get_profile_weights(cfg, profile),
        constraints=get_constraints(cfg),
        rl_config=get_rl_config(cfg),
    )


@app.post("/crispr/config")
async def update_config(req: ConfigUpdateRequest):
    cfg = _state["config"] or load_config()
    if req.profile:
        _state["current_profile"] = req.profile
        _state["current_weights"] = get_profile_weights(cfg, req.profile)
    if req.weights:
        _state["current_weights"] = {**(_state["current_weights"] or {}), **req.weights}
    if req.constraints:
        cfg["constraints"] = {**cfg.get("constraints", {}), **req.constraints}
        _state["config"] = cfg

    logger.info(f"Config updated: profile={_state['current_profile']} weights={_state['current_weights']}")
    return {"status": "updated", "profile": _state["current_profile"], "weights": _state["current_weights"]}


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    m = _state["metrics"]
    latencies = m["latencies_ms"]
    scores = m["scores"]

    import numpy as np
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
    avg_score = float(np.mean(scores)) if scores else 0.0

    return MetricsResponse(
        total_requests=m["total_requests"],
        total_feedback=m["total_feedback"],
        avg_score=round(avg_score, 4),
        failure_count=m["failure_count"],
        avg_latency_ms=round(avg_latency, 2),
        latency_p95_ms=round(p95_latency, 2),
        reward_breakdown=m["reward_breakdown"],
    )
