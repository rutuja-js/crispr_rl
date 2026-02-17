# crispr_rl

A production-ready CRISPR guide RNA design engine with reinforcement learning optimization, FastAPI service layer, and human-in-the-loop feedback system.

## Features

- **PAM Scanner**: Detect NGG (SpCas9) and custom PAM patterns on both strands
- **Feature Extraction**: GC content, thermodynamic proxy, run-length penalties, seed region scoring
- **Off-Target Filter**: Mismatch-tolerance search with seed-region weighting
- **Baseline Scorer**: Weighted heuristic scoring with configurable profiles
- **RL Optimizer**: Contextual bandit (epsilon-greedy + UCB) policy for guide selection
- **FastAPI Service**: RESTful API with human-in-the-loop feedback
- **SQLite Audit Log**: All model updates tracked with version history

## Quick Start

```bash
pip install -e .

# Run demo with a raw sequence
python demo/run_crispr_design.py --sequence ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG

# Run demo with gene IDs (uses simulated sequences)
python demo/run_crispr_design.py --gene_ids BRCA1 TP53 --profile knockout

# Start the API server
uvicorn crispr_rl.api.server:app --reload

# Run tests
pytest tests/ -v
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/crispr/design` | Design guide RNAs |
| POST | `/crispr/feedback` | Submit human feedback |
| GET | `/crispr/config` | Get current weights |
| POST | `/crispr/config` | Update weights |
| GET | `/crispr/ping` | Health check |
| GET | `/metrics` | Performance metrics |

## Profiles

- **knockout**: High specificity weight (NHEJ-mediated disruption)
- **knockdown**: High efficiency weight (partial reduction)
- **screening**: High coverage weight (library design)

## Biology Notes

- SpCas9 PAM: `5'-NGG-3'` downstream of 20nt protospacer
- Optimal GC: 40-60%
- Seed region: last 12nt before PAM (most critical for specificity)
- Off-target effects occur with ≤3 mismatches
