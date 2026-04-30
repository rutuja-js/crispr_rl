# crispr_rl — Review Packet

> Production-ready CRISPR guide RNA design engine with reinforcement learning optimization, FastAPI service layer, and human-in-the-loop feedback system.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Module-by-Module Review](#3-module-by-module-review)
   - [3.1 Feature Extraction](#31-feature-extraction)
   - [3.2 Scoring Layer](#32-scoring-layer)
   - [3.3 RL Core (Environment & Bandit)](#33-rl-core-environment--bandit)
   - [3.4 Trainer](#34-trainer)
   - [3.5 FastAPI Service](#35-fastapi-service)
   - [3.6 Schemas (Pydantic)](#36-schemas-pydantic)
   - [3.7 Python Client](#37-python-client)
   - [3.8 Evaluation Harness & Ablations](#38-evaluation-harness--ablations)
   - [3.9 Config & Utilities](#39-config--utilities)
4. [Data Flow — End-to-End Request Lifecycle](#4-data-flow--end-to-end-request-lifecycle)
5. [API Reference](#5-api-reference)
6. [Configuration Reference](#6-configuration-reference)
7. [Biology Domain Notes](#7-biology-domain-notes)
8. [Test Coverage Map](#8-test-coverage-map)
9. [Strengths](#9-strengths)
10. [Issues, Gaps & Recommendations](#10-issues-gaps--recommendations)
11. [Dependency Map](#11-dependency-map)

---

## 1. Project Overview

`crispr_rl` takes a raw DNA sequence (or gene IDs) and returns a ranked list of CRISPR guide RNA candidates, each scored by a multi-objective reward function and optimized by a contextual bandit RL policy.

**Key capabilities:**

- PAM site detection for SpCas9 (`NGG`) and configurable PAM patterns
- 5-dimensional feature extraction per guide: GC content, thermodynamic efficiency proxy, off-target specificity proxy, homopolymer penalty, and locus position
- Composite weighted reward with hard constraint enforcement (GC bounds, homopolymer run limits)
- Epsilon-greedy and UCB bandit policies with per-arm running statistics
- Pareto-front reranking across efficiency, specificity, and coverage dimensions
- FastAPI service with idempotent design requests, human feedback ingestion, and SQLite audit logging
- Sync + async Python client (`CRISPRClient`)
- Evaluation harness for baseline vs. RL comparison, plus component ablation studies

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                         FastAPI Server                        │
│  /crispr/design  /crispr/feedback  /crispr/config  /metrics  │
└────────────────────────┬────────────────┬────────────────────┘
                         │                │
              ┌──────────▼──────┐   ┌─────▼──────────┐
              │  PAM Scanner    │   │  SQLite DB      │
              │  (pam_scanner)  │   │  feedback +     │
              └──────────┬──────┘   │  audit_log      │
                         │          └─────────────────┘
              ┌──────────▼──────────────────────────┐
              │         Feature Extraction           │
              │  gc_content · thermo_proxy ·         │
              │  off_target · context · pam_scanner  │
              └──────────┬──────────────────────────┘
                         │
              ┌──────────▼──────────────────────────┐
              │         Scoring Layer                │
              │  baseline (heuristic rank) ──────┐  │
              │  composite_reward (weighted) ────┤  │
              │  pareto_rerank (multi-obj)   ────┘  │
              └──────────┬──────────────────────────┘
                         │
              ┌──────────▼──────────────────────────┐
              │           RL Core                    │
              │  GuideRNAEnv  ◄──── BanditTrainer   │
              │  EpsilonGreedyBandit / UCBBandit     │
              └──────────┬──────────────────────────┘
                         │
              ┌──────────▼──────────────────────────┐
              │         DesignResponse               │
              │  [GuideRNACandidate, …] + metadata   │
              └─────────────────────────────────────┘
```

**Layer responsibilities:**

| Layer | Files | Responsibility |
|---|---|---|
| API | `server.py`, `schemas.py` | HTTP routing, request validation, state management |
| Data | `loaders.py` | Sequence loading, FASTA parsing, RC computation |
| Features | `pam_scanner.py`, `gc_content.py`, `thermo_proxy.py`, `off_target.py`, `context.py` | Per-guide numerical features |
| Scoring | `baseline.py`, `composite.py` | Heuristic and RL-aware reward computation |
| RL | `env.py`, `bandit.py`, `trainer.py` | Gym-style environment, bandit policies, training loop |
| Eval | `harness.py`, `ablations.py`, `metrics_card.py` | Offline evaluation and component ablation |
| Utils | `config.py`, `logging.py`, `seeds.py` | Config loading, structured logging, reproducibility |
| Client | `client.py` | Sync + async HTTP client for the REST API |

---

## 3. Module-by-Module Review

### 3.1 Feature Extraction

#### `gc_content.py`
Computes two features:
- `gc_fraction(seq)` — fraction of G/C bases in the sequence, returned in `[0.0, 1.0]`
- `max_homopolymer_run(seq)` — length of the longest unbroken run of the same nucleotide

Both are used both as direct features in the RL state vector and as hard constraint guards in the scoring layer.

#### `thermo_proxy.py`
Provides `efficiency_proxy(seq)` — a lightweight thermodynamic approximation of guide RNA efficiency. This is a heuristic proxy (not a proper Tm or ΔG calculation) and is the main efficiency signal fed into the composite reward.

#### `off_target.py`
The most biologically rich module. Implements:
- `count_mismatches(guide, target)` — Hamming distance between equal-length sequences
- `weighted_mismatch_score(guide, target)` — mismatches in the 12-nt seed region (last 12 bases before PAM) are penalized 3× heavier than non-seed mismatches
- `find_off_targets(guide, sequence)` — scans both strands of the target sequence for windows with ≤3 mismatches
- `specificity_proxy(guide, sequence)` — returns `1 / (1 + Σ weighted_off_target_scores)`, excluding the on-target site
- `passes_specificity_filter(guide, sequence)` — hard threshold filter at 0.3

**Key constant:** `SEED_REGION_LENGTH = 12`, `SEED_WEIGHT = 3.0`, `SPECIFICITY_THRESHOLD = 0.3`

#### `pam_scanner.py`
Scans both strands of a DNA sequence for PAM-adjacent protospacers. Supports pattern matching for `NGG` (SpCas9), `TTTN` (Cas12a), `NNGRRT` (SaCas9), and custom patterns.

#### `context.py`
Provides `positional_score(position, seq_len)` — a coverage signal based on how centrally or evenly distributed a guide's position is in the full target sequence.

---

### 3.2 Scoring Layer

#### `baseline.py`
`rank_candidates(candidates, sequence, w1, w2, w3)` — applies the three-weight heuristic scoring to a candidate list in place. Weights correspond to efficiency (`w1`), specificity (`w2`), and coverage (`w3`). Used as the first-pass ranker before RL optimization.

#### `composite.py`
Two key functions:

**`composite_reward(guide_seq, sequence, position, weights, constraints)`**

```
reward = w_efficiency * efficiency_proxy
       + w_specificity * specificity_proxy
       + w_coverage * positional_score
```

Constraint violations (GC out of range, homopolymer run too long) apply a 0.5× soft penalty rather than a hard zero. Returns both the scalar reward and a `components` dict with individual scores and `risk_flags`.

**`pareto_rerank(candidates, objectives, max_candidates)`**

Applies Pareto dominance analysis across efficiency, specificity, and coverage. Candidates on the Pareto front are placed first; dominated candidates follow. Within each group, candidates are sorted by composite reward. This promotes diversity in the final output rather than collapsing to a single objective.

---

### 3.3 RL Core (Environment & Bandit)

#### `env.py` — `GuideRNAEnv`

A Gym-compatible environment where:
- **State**: flattened concatenation of 5-dim feature vectors for all candidates → shape `(n_candidates × 5,)`
- **Action**: integer index selecting one candidate from the pool
- **Reward**: `composite_reward` evaluated on the selected candidate

The 5 features per candidate are: `[gc_fraction, efficiency_proxy, specificity_proxy, max_homopolymer_run/20, position/seq_len]`

The environment resets automatically when all candidates have been selected (`done=True`), enabling multi-episode training over the same pool.

#### `bandit.py` — `EpsilonGreedyBandit` and `UCBBandit`

**EpsilonGreedyBandit:**
- Maintains per-arm running mean rewards (`values`) and pull counts (`counts`)
- Exploration decays: `ε = max(ε_end, ε * ε_decay)` after each update
- Defaults: `ε_start=0.3`, `ε_end=0.05`, `ε_decay=0.995`
- Context vector is accepted but not used (pure value-based, not contextual in the true sense)

**UCBBandit (UCB1):**
- UCB score: `mean_reward + c * sqrt(log(t) / n_pulls)`
- Unpulled arms are initialized first (round-robin) to ensure all arms are explored before exploitation begins
- Default `c = 1.414` (≈ √2, the standard UCB1 constant)

**Factory:** `make_bandit(policy, n_arms, config, seed)` constructs either from a config dict.

---

### 3.4 Trainer

#### `trainer.py` — `BanditTrainer`

Orchestrates the training loop:

1. Constructs `GuideRNAEnv` and a bandit via `make_bandit`
2. For each step: `select_arm` → `env.step` → `bandit.update`
3. Tracks best reward and best candidate seen across all steps
4. After training: re-evaluates all candidates with `composite_reward` and takes `max(rl_reward, composite_reward)` as the final score
5. Returns a `summary` dict with: `learning_curve`, `bandit_state`, `best_candidate`, `mean_reward`, `std_reward`, `latency_ms`

`top_k_candidates(k)` returns the top-k candidates by final bandit arm value.

Default: `n_steps=500`, `policy="epsilon_greedy"`.

---

### 3.5 FastAPI Service

#### `server.py`

**Startup (lifespan):** Loads YAML config, initializes `current_weights` from the `knockout` profile, and creates two SQLite tables (`feedback`, `audit_log`) via `aiosqlite`.

**Global mutable state (`_state`):** Config, current profile/weights, a request cache (for idempotency by `request_id`), and running metrics (request count, feedback count, latency list, score list, failure count).

**Request pipeline for `POST /crispr/design`:**

```
validate sequence / load gene sequence
  → _build_candidates (PAM scan + GC/homopolymer pre-filter)
  → rank_candidates (baseline heuristic)
  → _run_rl (BanditTrainer, n_steps=500)
  → pareto_rerank
  → serialize to DesignResponse
  → cache by request_id
```

**`POST /crispr/feedback`:** Stores rating (1–5) and notes to SQLite `feedback` table. Computes `delta = (rating - 3) / 2` as a normalized reward signal and writes to `audit_log`. Does **not** currently update the live bandit weights — this is a gap (see §10).

**`GET /metrics`:** Returns aggregate stats: total requests, total feedback, avg score, failure count, avg latency, p95 latency.

---

### 3.6 Schemas (Pydantic)

#### `schemas.py`

Includes a graceful fallback to Python `dataclasses` when Pydantic is not installed. The Pydantic path adds field-level validators:

- `GuideRNACandidate.seq`: must be exactly 20 nt, uppercase, ACGTN only
- `GuideRNACandidate.gc`: must be in `[0.0, 1.0]`
- `DesignRequest.sequence`: must be ≥23 nt (20 guide + 3 PAM), ACGTN only
- `FeedbackRequest.rating`: integer in `[1, 5]`

Profiles are validated as a `Literal["knockout", "knockdown", "screening"]`.

---

### 3.7 Python Client

#### `client.py` — `CRISPRClient`

Wraps every API endpoint with both sync (`httpx.Client`) and async (`httpx.AsyncClient`) versions:

| Method | Endpoint |
|---|---|
| `ping()` / (no async) | `GET /crispr/ping` |
| `design(**kwargs)` / `async_design(**kwargs)` | `POST /crispr/design` |
| `feedback(...)` / `async_feedback(...)` | `POST /crispr/feedback` |
| `get_config()` | `GET /crispr/config` |
| `update_config(...)` | `POST /crispr/config` |
| `get_metrics()` | `GET /metrics` |

Default base URL: `http://localhost:8000`. Default timeout: 60 seconds.

---

### 3.8 Evaluation Harness & Ablations

#### `harness.py` — `run_eval`

Runs baseline and RL pipelines on the same sequence and returns an `EvalResult` NamedTuple containing:
- `baseline_scores`, `rl_scores` — full score lists
- `baseline_mean`, `rl_mean` — aggregate means
- `uplift_pct` — `(rl_mean - baseline_mean) / baseline_mean * 100`
- `baseline_top5`, `rl_top5` — top candidate dicts
- `latency_ms_baseline`, `latency_ms_rl`

#### `ablations.py` — `run_ablations`

Three-condition study on a single sequence:
- `full` — all components active (default weights)
- `no_off_target` — `w_specificity=0`, `w_efficiency=0.7`, `w_coverage=0.3`
- `no_gc_context` — `w_specificity=0.2`, `w_efficiency=0.8`, `w_coverage=0.0`

Returns a dict mapping variant name → `EvalResult`.

---

### 3.9 Config & Utilities

#### `config.py`

Loads `default_config.yaml` (or a user-specified path). Provides three accessors:
- `get_profile_weights(config, profile)` — raises `ValueError` for unknown profiles
- `get_constraints(config)` — returns `gc_min`, `gc_max`, `max_homopolymer_run`, `max_candidates`, `seed_region_length`
- `get_rl_config(config)` — returns bandit hyperparameters

#### `default_config.yaml`

```yaml
profiles:
  knockout:    { w_specificity: 0.5, w_efficiency: 0.3, w_coverage: 0.2 }
  knockdown:   { w_specificity: 0.3, w_efficiency: 0.5, w_coverage: 0.2 }
  screening:   { w_specificity: 0.2, w_efficiency: 0.3, w_coverage: 0.5 }

constraints:
  max_homopolymer_run: 5
  gc_min: 0.35 / gc_max: 0.65
  max_candidates: 20
  seed_region_length: 12

rl:
  epsilon_start: 0.3 / epsilon_end: 0.05 / epsilon_decay: 0.995
  ucb_c: 1.414
  n_training_steps: 500
  seed: 42
```

#### `logging.py` / `seeds.py`

`get_logger` returns a structured logger tagged with `run_id`. `set_global_seed` seeds both NumPy and Python `random` for reproducibility.

---

## 4. Data Flow — End-to-End Request Lifecycle

```
Client POST /crispr/design
  { sequence | gene_ids, profile, seed, params }
         │
         ▼
  [Idempotency check] → return cached DesignResponse if request_id seen
         │
         ▼
  Sequence resolution
    • Direct sequence → validate_sequence (length ≥23, ACGTN only)
    • Gene IDs → load_gene_sequence (simulated) → concatenate with "N×10" separator
         │
         ▼
  _build_candidates
    • scan_sequence → PAM hits on both strands
    • Compute gc_fraction, max_homopolymer_run per hit
    • Flag constraint violations as risk_flags
    • Cap at max_candidates (default: 20)
         │
         ▼
  rank_candidates (baseline heuristic scorer)
    • Weighted sum: w_efficiency × eff + w_specificity × spec + w_coverage × cov
    • Sorts candidates in-place by score
         │
         ▼
  _run_rl → BanditTrainer.train(n_steps=500)
    • GuideRNAEnv.reset() → build state vector
    • For each step: select_arm → env.step → composite_reward → bandit.update
    • After training: re-score all candidates with composite_reward
    • Candidate score = max(bandit_arm_value, composite_reward)
         │
         ▼
  pareto_rerank
    • Identify Pareto-dominant candidates across efficiency/specificity/coverage
    • Place Pareto front first, sort each tier by reward
         │
         ▼
  Serialize → DesignResponse
    { request_id, run_id, candidates: [GuideRNACandidate×N], metadata }
         │
         ▼
  Cache + return to client
```

---

## 5. API Reference

### `POST /crispr/design`

**Request:**
```json
{
  "request_id": "optional-uuid",
  "sequence": "ATCG...(≥23 nt)",
  "gene_ids": ["BRCA1", "TP53"],
  "organism": "human",
  "profile": "knockout | knockdown | screening",
  "seed": 42,
  "params": { "pam_pattern": "NGG" }
}
```
Provide either `sequence` or `gene_ids`, not both.

**Response:** `DesignResponse`
```json
{
  "request_id": "...",
  "run_id": "...",
  "candidates": [
    {
      "id": "guide_abc123_000",
      "seq": "ATCGATCGATCGATCGATCG",
      "pam": "NGG",
      "locus": "pos:42",
      "strand": "+",
      "gc": 0.50,
      "features": { "gc": 0.50, "max_homopolymer": 2, ... },
      "score": 0.7231,
      "explanations": { "efficiency": 0.72, "specificity": 0.81, "coverage": 0.55 },
      "risk_flags": []
    }
  ],
  "metadata": { "profile": "knockout", "weights": {...}, "latency_ms": 312.5 }
}
```

### `POST /crispr/feedback`

```json
{ "candidate_id": "guide_abc123_000", "rating": 4, "notes": "...", "rationale": "..." }
```
Rating scale: 1 (poor) to 5 (excellent). Stored in SQLite; `delta_reward = (rating - 3) / 2`.

### `GET /crispr/config`

Returns current profile, weights, constraints, and RL config.

### `POST /crispr/config`

```json
{ "profile": "knockdown", "weights": { "w_efficiency": 0.6 } }
```

### `GET /metrics`

Returns: `total_requests`, `total_feedback`, `avg_score`, `failure_count`, `avg_latency_ms`, `latency_p95_ms`.

### `GET /crispr/ping`

Health check → `{ "status": "ok" }`.

---

## 6. Configuration Reference

| Key | Default | Description |
|---|---|---|
| `profiles.knockout.w_specificity` | 0.5 | Weight on off-target specificity |
| `profiles.knockout.w_efficiency` | 0.3 | Weight on thermodynamic efficiency |
| `profiles.knockout.w_coverage` | 0.2 | Weight on locus coverage |
| `constraints.gc_min` | 0.35 | Minimum acceptable GC fraction |
| `constraints.gc_max` | 0.65 | Maximum acceptable GC fraction |
| `constraints.max_homopolymer_run` | 5 | Max allowed same-base run length |
| `constraints.max_candidates` | 20 | Max candidates returned |
| `constraints.seed_region_length` | 12 | Seed region length (nt before PAM) |
| `rl.epsilon_start` | 0.3 | Initial exploration rate |
| `rl.epsilon_end` | 0.05 | Minimum exploration rate |
| `rl.epsilon_decay` | 0.995 | Per-step epsilon decay factor |
| `rl.ucb_c` | 1.414 | UCB confidence interval constant |
| `rl.n_training_steps` | 500 | Bandit training steps per request |
| `rl.seed` | 42 | Global RNG seed |

**Note:** The `README.md` shows different weight values for profiles than the actual `default_config.yaml`. For example, the README lists "knockout: high specificity weight" with `w_efficiency=0.5, w_specificity=0.3` but the YAML sets `w_specificity=0.5, w_efficiency=0.3`. Trust the YAML.

---

## 7. Biology Domain Notes

| Concept | Value | Source |
|---|---|---|
| SpCas9 PAM | `5'-NGG-3'` downstream of protospacer | `pam_scanner.py`, `default_config.yaml` |
| Guide RNA length | 20 nt | `schemas.py` validator |
| Optimal GC | 40–65% (`gc_min=0.35`) | Empirical; `default_config.yaml` |
| Seed region | Last 12 nt before PAM | Most critical for specificity; `off_target.py` |
| Off-target threshold | ≤3 mismatches | `off_target.py` `max_mismatches` default |
| Seed weight | 3× heavier than non-seed | `SEED_WEIGHT=3.0` in `off_target.py` |
| Profiles | `knockout` (high specificity), `knockdown` (high efficiency), `screening` (high coverage) | `default_config.yaml` |

The `thermo_proxy` is explicitly a heuristic approximation — it does not compute actual Tm or ΔG values. For production bioinformatics use, this should be replaced with a proper thermodynamics library (e.g., `primer3`, `ViennaRNA`).

---

## 8. Test Coverage Map

| Test File | Covers |
|---|---|
| `test_pam_scanner.py` | PAM detection on both strands, edge cases |
| `test_features.py` | GC fraction, homopolymer run, thermo proxy |
| `test_off_target.py` | Mismatch counting, weighted scoring, specificity proxy |
| `test_baseline_scorer.py` | Weighted heuristic ranking |
| `test_rl_env.py` | `GuideRNAEnv` state/action/reward mechanics |
| `test_api_schemas.py` | Pydantic validators for all schema classes |
| `test_integration.py` | End-to-end pipeline with FastAPI test client |

No dedicated unit tests found for:
- `composite.py` (Pareto reranking logic)
- `bandit.py` (epsilon decay, UCB score correctness)
- `trainer.py` (training loop behavior, top-k output)
- `client.py` (HTTP client, error handling)
- `harness.py` / `ablations.py`

---

## 9. Strengths

**Clean separation of concerns.** Feature extraction, scoring, RL, and API layers are well-isolated with minimal cross-layer coupling. Each module has a clear, single responsibility.

**Graceful degradation.** The Pydantic fallback to dataclasses in `schemas.py` means the core logic works even in minimal environments.

**Idempotent API.** The `request_id` cache in `server.py` prevents duplicate design runs on retry, which is important for expensive RL training calls.

**Pareto-aware diversity.** Using Pareto reranking rather than pure scalar sorting ensures the returned candidate set spans multiple objectives — practically important when a biologist wants guides that differ in their efficiency/specificity trade-offs.

**Structured audit trail.** The SQLite `audit_log` table records every feedback event with a version UUID, enabling model version tracking and delta reward history.

**Reproducibility.** `seeds.py` + per-request `seed` parameter in `DesignRequest` make runs fully reproducible.

**Good biology grounding.** Seed-region weighting in off-target scoring, strand-aware PAM scanning, and profile weights that map to real experimental use cases (NHEJ disruption vs. partial knockdown vs. library screening) reflect genuine CRISPR biology.

---

## 10. Issues, Gaps & Recommendations

### Critical

**1. Feedback loop is not wired to the live model.**
`POST /crispr/feedback` stores ratings to SQLite and computes `delta_reward`, but never feeds this signal back to update `_state["current_weights"]` or the bandit values. The human-in-the-loop story is incomplete.

*Recommendation:* Add a weight-update step in the feedback handler: `new_weight = old_weight + lr * delta_reward` (clipped), and optionally persist updated weights to config.

**2. In-memory request cache grows unboundedly.**
`_state["request_cache"]` is a plain dict with no eviction policy. Under production load, this will exhaust memory.

*Recommendation:* Replace with an LRU cache (`functools.lru_cache` or `cachetools.LRUCache`) capped at a reasonable size (e.g., 1,000 entries), or use Redis.

**3. Bandit context is unused.**
Both `EpsilonGreedyBandit.select_arm` and `UCBBandit.select_arm` accept a `context` parameter but ignore it. The system is labeled "contextual bandit" but operates as a standard multi-armed bandit.

*Recommendation:* Either implement a true contextual policy (e.g., LinUCB using the 5-dim feature vector) or remove the `context` parameter to avoid misleading documentation.

### Moderate

**4. `thermo_proxy` is a placeholder.**
The efficiency proxy does not compute real thermodynamic values. This is the largest scientific gap in the feature set.

*Recommendation:* Integrate `primer3-py` for proper Tm calculation, or use a published guide RNA scoring model (e.g., Rule Set 2, DeepCRISPR outputs).

**5. Off-target search is O(n²) naive scan.**
`find_off_targets` slides a window over the entire sequence + its RC, making it O(L × G) per guide. For real genomic sequences (millions of bp), this is unusable.

*Recommendation:* Use a seed-and-extend approach, or integrate an existing off-target tool (e.g., Cas-OFFinder, CRISPOR) as a subprocess or library call.

**6. Gene sequence loading is simulated.**
`load_gene_sequence` returns fake/random sequences. Real gene IDs are not resolved from any database.

*Recommendation:* Add optional Entrez/NCBI fetch via `biopython` (already listed as an optional dependency) or accept a FASTA input endpoint.

**7. README weight values differ from config.**
The README states `knockout` has `w_efficiency=0.5, w_specificity=0.3` but the YAML sets `w_specificity=0.5, w_efficiency=0.3`. This creates confusion for new contributors.

*Recommendation:* Auto-generate the README profile table from the YAML at documentation build time.

### Minor

**8. No rate limiting or authentication on the API.**
The server has no API key, JWT, or rate-limiting middleware.

**9. `_state` is not thread-safe.**
Concurrent writes to the metrics dict and request cache are not protected by locks. FastAPI runs async, so this is lower risk, but CPU-bound RL training in `_run_rl` is called synchronously (blocking the event loop).

*Recommendation:* Move `BanditTrainer.train` into a thread pool via `asyncio.run_in_executor`.

**10. `.cpython-312.pyc` files are committed to git.**
The `__pycache__` artifacts should be in `.gitignore`.

---

## 11. Dependency Map

```
crispr_rl (core)
├── numpy ≥1.24
├── pandas ≥2.0
└── pyyaml

crispr_rl[api]
├── fastapi ≥0.110
├── uvicorn[standard]
├── pydantic ≥2.0
├── aiosqlite
└── httpx

crispr_rl[bio]  (optional)
└── biopython

crispr_rl[test]
├── pytest
└── pytest-asyncio

crispr_rl[dev]
├── black
├── ruff
└── mypy
```

Python ≥3.10 required (uses `match` syntax in some paths, `X | Y` union types).

---

*Review packet generated from source at: `github.com/rutuja-js/crispr_rl`*
*Files reviewed: env.py, bandit.py, trainer.py, server.py, schemas.py, client.py, composite.py, off_target.py, baseline.py, gc_content.py, thermo_proxy.py, pam_scanner.py, context.py, harness.py, ablations.py, config.py, default_config.yaml, pyproject.toml, README.md*
