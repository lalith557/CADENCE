# CADENCE

**Causal Attribution-Driven Efficient Continual Retraining for Production ML Systems**

CADENCE watches deployed ML models, tells you **which upstream factor caused a performance drop**, and automatically picks the **minimal retraining scope** that restores performance — saving compute and carbon vs. periodic-retrain or reactive-full-retrain approaches.

The full research proposal, walkthrough example, and dataset reference live in [`CADENCE-master-reference.md`](CADENCE-master-reference.md). Living documents track the actual build:

- [`docs/STATUS.md`](docs/STATUS.md) — **honest end-of-project ledger**; start here
- [`docs/decisions.md`](docs/decisions.md) — every real design choice, with reasoning + alternatives
- [`docs/tech.md`](docs/tech.md) — stack rationale
- [`docs/prior_art.md`](docs/prior_art.md) — related-work notes per claim
- [`docs/results.md`](docs/results.md) — measured experiment results (no fabrications)
- [`docs/weaknesses.md`](docs/weaknesses.md) — self-audit register
- [`docs/product.md`](docs/product.md) — one-page launchability pitch
- [`docs/paper/abstract.md`](docs/paper/abstract.md) — draft abstract + method skeleton
- [`docs/paper/skeleton.md`](docs/paper/skeleton.md) — auto-generated results appendix

---

## Architecture

```
Live Traffic → Inference Gateway → predictions/logs
       │
       ▼
Streaming Feature + Activation Collector  (sampled, privacy-filtered)
       │
       ▼
CDAG Builder (PC + NOTEARS) + GNN encoder
       │
       ▼
Counterfactual Attribution (learned surrogate) ─▶ ranked responsibility scores
       │
       ▼
Retraining Scope Optimizer (PPO, cost + carbon + SLA constrained)
       │
       ▼
Continual Retraining Executor (EWC-regularized partial / full retrain)
       │
       ▼
Shadow / Canary → promote or roll back → resume monitoring
```

Four trained ML components: **FraudNet** (or user-supplied model), **CDAG-GNN**, **counterfactual surrogate**, **PPO policy**. Plus a re-fit-each-cycle causal-discovery procedure (hybrid PC + NOTEARS) and EWC as a regularizer on the production model. See [`CADENCE-master-reference.md`](CADENCE-master-reference.md) Part 4 for the exact count.

---

## Quickstart (5 minutes)

Verified on a clean Python 3.13 venv on 2026-08-30 — see `docs/STATUS.md`.

```bash
# 1. Set up a Python 3.10-3.13 virtualenv
python -m venv .venv
source .venv/bin/activate   # or: .venv\Scripts\activate  on Windows

# 2. Install core + dev deps
pip install -e .[dev]

# 3. Run the test suite (skips tests that need Kaggle CSVs)
pytest -m "not needs_dataset and not gpu and not slow"

# 4. Smoke-check the pipeline: config → seeds → MLflow round-trip
cadence smoke -c configs/ci.yaml

# 5. Prove GPU is really used (matmul + peak VRAM report)
cadence gpu-check
```

Product surface:

```bash
# Dashboard: real CDAG + real responsibility from experiments/*.json
streamlit run dashboard/app.py

# REST API: /health /score /monitor /decide /admin/reload
uvicorn cadence.api.server:app --port 8000

# Or the whole stack under compose (needs a live Docker daemon)
docker compose up --build dashboard mlflow
```

To enable the full ML stack (Phase 1+):

```bash
pip install -e .[ml,drift]         # + torch, SB3, river, CodeCarbon
pip install -e .[gnn]              # + torch-geometric  (Phase 3)
pip install -e .[causal]           # + causal-learn, gcastle  (Phase 3)
pip install -e .[cl]               # + avalanche-lib  (Phase 6)
pip install -e .[api]              # + fastapi, streamlit  (Phase 7)
# or install everything:
pip install -e .[all]
```

To verify local datasets are in place (see [`CADENCE-datasets-verified.md`](CADENCE-datasets-verified.md) for where to download):

```bash
cadence datasets-check
```

---

## Hardware assumptions

- **Reference target:** GTX 1650, 4 GB VRAM. All numbers in `docs/results.md` come from configs sized to fit this budget — see `configs/default.yaml`.
- **Local dev:** the machine used here has an RTX 4070 (8 GB). `configs/local.yaml` allows the extra headroom, **but any run whose numbers land in the paper must use `default.yaml` or `ci.yaml`.** This invariant is enforced by the config validator.

Heavy experiments (CIFAR-10 full, Criteo/Avazu) are Colab/Kaggle-only per the reference.

---

## Repo layout

```
cadence/
  common/         config, logging, seeding, MLflow wrapper
  data/           dataset loaders (Fraud, GMSC, Telco, Adult, Elec2, MNIST)
  adapters/       ModelAdapter interface: neural / tree / linear
  collector/      streaming feature + activation collection
  cdag/           causal graph builder (PC + NOTEARS) + node/cluster construction
  attribution/    GNN + counterfactual surrogate + responsibility scoring
  rso/            PPO-based Retraining Scope Optimizer + sandbox Gym env
  executor/       EWC-regularized partial retrain + full retrain + shadow/canary
  carbon/         CodeCarbon integration + Electricity Maps client
  api/            FastAPI dashboard backend
benchmarks/       synthetic drift generator + replayed-incident harnesses
tests/            unit + integration
docker/           Dockerfile(s)
configs/          default.yaml, local.yaml, ci.yaml
docs/             living documents + paper skeleton
experiments/      MLflow tracking store (default)
Dataset/          Kaggle/UCI CSVs (not committed — see .gitignore)
```

---

## Development

```bash
make setup       # core + dev deps
make setup-ml    # + torch, SB3, river, CodeCarbon
make test        # full test suite (skips needs_dataset)
make lint        # ruff check
make format      # ruff format
make typecheck   # mypy (advisory in Phase 0)
make smoke       # CLI end-to-end smoke
```

---

## Status

**Phase 0 — Scaffold** ✅ done. See [`docs/results.md`](docs/results.md) R-0.
**Phase 1 — FraudNet + drift + baselines** ✅ done. R-1..R-3.
Next: **Phase 2** — RSO/PPO, cost/carbon sensitivity sweep, H2 test at n=10 × 5 scenarios.

The full phase plan lives in [`docs/phase-plan.md`](docs/phase-plan.md).

---

## Citation

Once a paper is out, cite as:

```
Lalith Gona. CADENCE: Causal Attribution-Driven Efficient Continual Retraining
for Production ML Systems. Preprint, 2026.
```

## License

Apache-2.0. See [LICENSE](LICENSE).
