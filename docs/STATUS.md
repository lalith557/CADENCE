# CADENCE — Final Status (2026-08-30)

This file is the honest end-of-project ledger. If it disagrees with
`README.md` / `product.md` / a slide deck, `STATUS.md` wins.

## One-paragraph summary

CADENCE is a working end-to-end MLOps loop that detects drift, attributes
it to specific features + activation clusters via a GraphSAGE-style GNN
over a NOTEARS-fitted causal graph, predicts recovery per candidate via a
jointly-trained surrogate, and picks a retraining scope (no-op / partial /
full) via a learned PPO policy with a cost + carbon + SLA-constrained
reward. All eight §4 fallbacks are implemented and tested. The pipeline
runs on GPU (RTX 4070 Laptop verified). H1 (GNN > PSI) and H3 (partial EWC
reduces forgetting) are supported at paper scale with paired Wilcoxon
p < 0.05; H2 (cost-win) is a Pareto position on real drift but not a strict
dominance. H2 at the full 10-seed × 30k-timestep Gate A protocol is not
runnable in a single-session laptop budget — a resume-able Colab driver
ships in `scripts/colab_gate_a.py` for the paper-grade run.

## Anchor claim scoreboard

| Claim | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| A | CDAG + GNN attribution beats correlational PSI | **SUPPORTED (paper-scale)** | R-Gate-B at n=9 hard subset p=0.016; R-Gate-B-n10 (this session) at n=50, per easy/hard subset breakdown, p in results.md |
| B | Learned surrogate predicts post-fix F1 for a node without running the intervention | **SUPPORTED (calibration)** | R-Gate-C: joint train val R²=0.888, MAE=0.036 on 90 triples. Train/test gap on unrecoverable OOD scenarios recorded honestly. |
| C — flagship | RSO gives a strict cost win over baselines under a cost + carbon + SLA reward | **PARETO ONLY on this session's compute budget** | R-Gate-E Elec2: 69% GPU-hr saved vs reactive_full at −1.7% F1. Not a strict dominance. Full 10-seed × 30k Gate A run pending Colab (`scripts/colab_gate_a.py`). |
| H3 | Partial EWC retrain has less forgetting than naive full | **SUPPORTED (paper-scale)** | R-Gate-F-mnist-n10: partial forgetting −0.032 vs full +0.395 vs full+replay −0.017. Wilcoxon vs both baselines p = 0.001 (n=10). |

## What ships (product surface)

- **Streamlit dashboard** — real NOTEARS-weighted CDAG rendering + real
  per-feature responsibility bars + real decision + real cost saved.
  Falls back to illustrative panels only for pre-W-30 artifacts.
- **FastAPI REST server** (`cadence/api/server.py`) — `/health`, `/score`,
  `/monitor`, `/decide`, `/admin/reload`. `/decide` runs the learned PPO
  when loaded, §4.6 false-alarm suppression, rule fallback otherwise.
- **docker-compose stack** — dashboard + MLflow (sqlite backend) services.
  Compose config syntactically clean; boot pending Docker daemon (W-31).
- **CLI**: `cadence gpu-check`, `cadence smoke`, `cadence datasets-check`,
  `cadence config-show`.
- **CI**: `.github/workflows/ci.yml` runs lint + tests + docker-build +
  fresh-venv install on push. Never executed against GitHub because the
  repo isn't pushed (W-8 open on the "clone from GitHub and go" front).
- **Test suite**: 82 tests green (75 pre-existing + 6 API + 1 e2e).
- **Paper skeleton**: auto-generated from `docs/results.md`, ~18 R-entries
  with tables and Wilcoxon annotations.

## What does NOT ship (honest gaps)

- **Full-scale Gate A (H2)** paired-Wilcoxon Pareto verdict. Requires
  10 seeds × 30 k PPO timesteps × 5+ scenarios. Local single-seed
  800-timestep run already took 2h45m; the full sweep would be days.
  The Colab driver is written and self-checkpoints per seed — the run
  is *ready*, just not *done* in this session.
- **Docker-compose actually booted** — Docker Desktop daemon not running on
  this machine. Compose file validated; images not built.
- **GitHub Actions CI never green on a real push** — workflow written, never
  executed remotely.
- **Continuous / hybrid action space** for the RSO — the plan's W-27 fix
  (4) is not implemented; PPO stays on Discrete(3). Everything else in
  W-27 (longer training, contested-SLA scenarios, richer state,
  Aug-Lagrangian) shipped.

## Where each anchor's evidence lives

- `docs/results.md` R-Gate-B, R-Gate-B-n10 (H1)
- `docs/results.md` R-Gate-C (H2 closed-loop plumbing + surrogate calibration)
- `docs/results.md` R-Gate-D (H3 negative on single-task Fraud — kept as
  honest limitation), R-Gate-F-mnist-n10 (H3 positive on multi-task MNIST)
- `docs/results.md` R-Gate-E (real-drift outcome grading on Elec2/Airlines)
- `docs/results.md` R-Gate-F, R-Gate-F-text (generality tabular/tree/text)
- `docs/results.md` R-Gate-A-w28 (PPO retrained on 15-d obs; drives the RSO)
- `docs/results.md` R-Gate-G, R-Gate-G-clean-clone (product surface + audit)
- `experiments/*.json` — per-run raw artifacts every R-entry points at

## Weakness register status

`docs/weaknesses.md` has 32 W-entries. As of this session:

- **FIXED** (with commit + R-entry annotation): W-16, W-18, W-19, W-21,
  W-22, W-23, W-24, W-26, W-28 (engineering close-out).
- **PARTIAL FIX**: W-27 (fixes 1/2/3/5 landed; continuous action space
  deferred), W-29 (Yelp domain-shift landed; Amazon Reviews 2023 not
  used), W-31 (compose config OK; runtime boot pending Docker daemon).
- **OPEN, blocking**: none.
- **OPEN, deferred with justification**: W-4 (prior-art done at
  first-pass; deep-read follow-ups deferred), W-8 (CI needs GitHub push),
  W-9 (Docker needs live daemon), W-11 (mypy still advisory),
  W-15/W-30-followups (cosmetic).

## Runbook — how to reproduce every R-entry

```bash
# GPU + install
make setup-all
cadence gpu-check

# H1 (paper scale)
python -m benchmarks.phase_b_run --seeds 10 --gnn-samples 200 --gnn-epochs 20

# H3 (paper scale)
python -m benchmarks.phase_f_mnist --seeds 10 --ewc-penalty 5000

# H2 flagship (paper scale — needs Colab; see scripts/colab_gate_a.md)
python scripts/colab_gate_a.py --seeds 10 --train-timesteps 30000

# Real drift end-to-end
python -m benchmarks.phase_e_run --dataset elec2 --seeds 10
python -m benchmarks.phase_e_run --dataset airlines --seeds 10

# Generality
python -m benchmarks.phase_f_tree --seeds 5 --sla 0.45
python -m benchmarks.phase_f_text --seeds 3 --split domain

# Dashboard + API
streamlit run dashboard/app.py       # or docker-compose up dashboard mlflow
uvicorn cadence.api.server:app --port 8000
```
