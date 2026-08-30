# CADENCE — Paper draft: abstract, intro, method skeleton

*Living document. Numbers below come from `docs/results.md`; regenerate
`docs/paper/skeleton.md` after every results update.*

---

## Abstract (draft, 2026-08-30)

Production machine-learning systems degrade continuously as their input
distributions drift, yet no widely deployed monitoring tool tells engineers
*which* upstream factor is responsible for a specific performance drop, and
no MLOps stack decides *how much* of the model to retrain in response.
CADENCE closes this loop with three coupled contributions: (1) a **Causal
Drift Attribution Graph (CDAG)** that unifies external feature nodes and
internal activation-cluster nodes into a single incrementally-updated
weighted DAG; (2) a **counterfactual surrogate** trained jointly with a
GraphSAGE-style GNN over the CDAG to predict per-node post-fix F1 without
running the intervention; and (3) a **Retraining Scope Optimizer (RSO)** — a
PPO policy over `{no-op, partial-subnetwork-retrain, full-retrain}` with a
compute + carbon + SLA-constrained augmented-Lagrangian reward. Every stage
runs on GPU; every §4 fallback (shadow validation, rollback, escalation
ladder, unrecoverable-drift guard, diffuse-responsibility fallback,
false-alarm suppression, resource fallback, cold-start degradation) is
implemented and tested. On the standard synthetic drift benchmark the
GNN-based CDAG attribution significantly out-performs correlational PSI at
n=10 seeds with paired Wilcoxon p < 0.05; on Split-MNIST {0,1} → {2,3} the
EWC-regularised partial retrain produces −3 % Task-A forgetting vs +40 %
under naive full retrain and −2 % under full-with-replay, paired Wilcoxon
p < 0.001 vs both baselines; on real drift (Elec2) CADENCE saves 69 % of
GPU-hours vs a reactive-full baseline at a 1.7 % F1 cost — a Pareto-frontier
option, not a strict dominance. We ship an end-to-end open-source
implementation with a Streamlit dashboard rendering the actual CDAG, a
FastAPI REST surface, dockerised deployment, and honest limitations
recorded per gate.

---

## 1. Introduction

Every deployed ML model degrades over time as the world drifts away from
its training distribution. Industry surveys consistently name drift as one
of the top production-failure modes, and most teams either (a) retrain on
a fixed schedule regardless of whether drift has occurred, wasting compute
and carbon, or (b) wait for a human to notice a KPI drop, by which point
real damage has already occurred. A large production-model retrain can
cost thousands to millions of dollars in compute, so *when*, *what*, and
*how much* to retrain is itself a high-stakes, poorly-automated decision.
And no widely deployed system today can tell an engineer *why* a model
broke — only *that* something drifted.

Prior work spans four communities, none of which closes the loop. **Statistical
drift detectors** (PSI, KS, JS) tell you that a distribution moved but not
whether it *caused* the drop, driving high false-alarm rates and wasted
retrains. **Causal-discovery for model debugging** (PC, NOTEARS, concurrent
2025 work in TempXAI) builds feature-space graphs but does not reach into
model activations and does not drive the retraining decision. **Continual
learning** (EWC, LwF, replay-based methods) protects against forgetting
during retraining but assumes the decision to retrain has already been
made. **Cost-aware MLOps** (retrain-on-schedule frameworks; Databricks /
SageMaker automation) treats retrain *timing* as a decision but never
retrain *scope*.

CADENCE's contribution is the specific triple of (a) a unified causal
graph over feature nodes AND internal activation-cluster nodes, (b) a
learned surrogate that scores per-node responsibility without running
ground-truth interventions, and (c) a cost-and-carbon-aware RL policy that
chooses retraining *scope* rather than timing. The closed-loop coupling of
attribution → scope decision is what none of the surveyed prior work
occupies.

## 2. Method (skeleton — see `cadence/` module docstrings for details)

- 2.1 System architecture — the loop described above, wired through a
  `ModelAdapter` protocol so the same code drives FraudNet, LightGBM,
  and TF-IDF+LR adapters unchanged.
- 2.2 CDAG construction — PC (skeleton) + NOTEARS (continuous weights)
  over per-window statistics of feature and activation-cluster nodes.
- 2.3 GNN + surrogate — 2-layer GCNConv over the CDAG on GPU, jointly
  trained end-to-end with an MLP surrogate that predicts post-fix F1
  from `(node embedding, pre_F1, severity)`.
- 2.4 RSO — PPO on a 15-dim observation with an Augmented-Lagrangian
  dual-λ update; action ∈ {no-op, partial, full}; reward = ΔF1 −
  w_gpu · compute − w_co2 · carbon − λ · SLA_violation.
- 2.5 Executor + §4 fallbacks — Part-3B EWC partial retrain with
  shadow validation and rollback; escalation ladder partial → full →
  human; unrecoverable-drift guard; diffuse-responsibility escalation;
  false-alarm suppression; resource fallback; cold-start degradation.

## 3. Experiments — see `docs/paper/skeleton.md`

Auto-generated per-entry tables live in the skeleton file; the sections
below wrap them into the paper narrative.

- 3.1 Attribution (H1) on Credit Card Fraud + synthetic injections.
- 3.2 Forgetting (H3) on Split-MNIST + Fraud (single-task negative).
- 3.3 Real-drift end-to-end (H2 outcome-graded) on Elec2 + Airlines.
- 3.4 Generality — LightGBM on GMSC, TF-IDF+LR on Yelp domain shift.
- 3.5 Robustness — PSI false-alarm rate, GNN AUROC under edge dropout.
- 3.6 The RSO flagship (Claim C, full Gate A protocol) — see
  `scripts/colab_gate_a.md` for the paper-scale run; the R-Gate-A-w28
  entry documents the smoke-scale close-out and honestly flags what
  the paper-scale run must add.

## 4. Honest limitations

- The full-scale RSO Pareto claim (H2) is not proven strictly in this
  submission's session-scoped runs; a resume-able Colab notebook ships
  the paper-protocol config and is the right medium for the multi-day
  run.
- The single-task benchmark (Fraud) does not support the classical
  forgetting claim, and we report that negative honestly. The
  multi-task benchmark (Split-MNIST) does; both are documented.
- The dashboard uses illustrative CDAG panels only for pre-W-30
  artifacts; every new run ships the raw NOTEARS weights.
- Text-drift generality tests plumbing (TF-IDF+LR) but does not push
  the classifier below SLA at this sample size; a paper-grade text
  experiment needs Amazon Reviews 2023 at ~1M reviews.

## 5. Reproducibility

- Every R-entry cites the exact CLI to reproduce.
- Every `experiments/*.json` artifact is committed alongside the R-entry.
- `cadence gpu-check` + `make test` gate every gate.
- `scripts/colab_gate_a.py` gives the paper-protocol H2 run in an
  afternoon on a T4 / A100.
