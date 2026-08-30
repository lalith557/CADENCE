# CADENCE — Product Positioning (one page)

## Who it's for
Mid-to-large ML platform teams running many production models where compute + retraining budgets are visible on someone's P&L: **fintech, ad-tech, health-tech, e-commerce, insurance**. Teams with two symptoms:
1. They retrain on a cron schedule "just to be safe" and pay for compute they don't need.
2. When a model degrades, no one can point to *which upstream factor* is causing the drop — only that something drifted.

## What it does
CADENCE watches a deployed ML model in production and, when it starts to degrade, tells you **why** — pointing to specific upstream features or internal subnetworks — and then automatically chooses the **minimal** retraining scope that will restore performance, saving compute and carbon.

## The differentiator vs. Evidently / WhyLabs / Arize / Fiddler
Every incumbent does **statistical drift detection** — "PSI on feature X crossed a threshold." None of them:
- Reach *inside the model* to attribute drift to specific activation clusters.
- Frame retraining scope as a *learned decision* rather than a fixed rule ("if PSI > 0.25, retrain everything").
- Optimize for compute + carbon in the same loop.

The moat is the **causal attribution algorithm** (patent-pending, Claim A + B in the reference proposal) and the **cost/carbon-aware retraining policy** (Claim C).

## The pitch
> "We pay for ourselves in the first quarter by cutting your retraining GPU-hours 40–70%, and while we're at it, we tell you *why* your models broke."

## Business model
Two tiers, both usage-metered:
- **Team**: SaaS, priced per monitored model per month, self-serve dashboard.
- **Enterprise**: on-prem/VPC deployment (regulated industries), priced as **a percentage of GPU-hours saved vs. their baseline**. Zero-risk ROI framing for the buyer.

## Launchable MVP scope (what "v1" is)
Enough to sign the first design partner:
1. Model Adapter interface with adapters for PyTorch neural nets and LightGBM tree ensembles (covers ~80% of fintech/ad-tech production models).
2. CDAG + attribution running as a scheduled job (not yet real-time streaming — that's v2).
3. RSO recommending scope; **human-in-the-loop approval** on the retrain decision (auto-apply is v2).
4. Dashboard: live drift, CDAG, responsibility scores, retraining recommendation, cost/carbon saved.
5. Slack + PagerDuty webhook on alerts.
6. Audit log of every decision (compliance sell).

## What is explicitly **not** in the MVP
- Multi-modal (text, image) — v2.
- Federated / edge deployments — v3.
- LLM behavioral drift — the research direction, not the product.

## Why we can build it now
- The paper's four novelty pieces (CDAG, GNN, surrogate, RSO) are all engineerable in 2026 with off-the-shelf primitives (PyG, SB3, causal-learn).
- The mainstream MLOps monitoring market is now three years old — buyers understand the category, but the incumbents haven't moved from correlation to causation. That window is what we're jumping through.

---

## What's actually shipped, gate-by-gate (as of 2026-08-30)

Every claim below points at a live entry in `docs/results.md` with the exact
command that produced its numbers.

- **Gate 1 (GPU)** — `cadence gpu-check` prints RTX 4070 Laptop + 13.05 measured
  TFLOPs; every training run logs peak VRAM to MLflow. Carbon profile fixed
  from the wrong `gtx-1650` default.
- **Step A (RSO rescue)** — contested-SLA scenarios, EWC-in-sandbox, Augmented-
  Lagrangian dual-λ, enriched 15-d observation. Wiring proven end-to-end
  (R-Gate-A-smoke); full 30k-timestep evidence run queued.
- **Step B (H1 / Claim A)** — GraphSAGE-style GNN over the CDAG on cuda beats
  correlational PSI at α=0.05: MRR/AUROC paired Wilcoxon p = 0.0156 vs PSI,
  p = 0.021–0.029 vs the pre-Step-B structural proxy.
- **Step C (Claim B / closed loop)** — surrogate + joint GNN training on GPU;
  one command runs pretrain → CDAG → GNN → surrogate → RSO → executor →
  validation. Val R² = 0.888 on 90 intervention triples in 6.4 s / 65 MB VRAM.
- **Step D (executor + fallbacks)** — Part-3B EWC executor + all 8 §4 fallbacks
  (shadow, rollback, escalation ladder, unrecoverable guard, diffuse,
  false-alarm, resource, cold-start). 21 tests, each fallback forced and
  verified. Gate D H3 honest **negative** on Fraud (single-task limitation).
- **Step E (real drift)** — Elec2 + Airlines end-to-end. On Elec2 CADENCE saves
  **69 % compute vs reactive_full at 1.7 % F1 loss**; strict-dominates neither
  baseline but sits on the Pareto frontier. Airlines shows PSI silent on
  hashed categoricals — honest limit recorded.
- **Step F (generality + H3 rerun)** — LightGBM tree adapter on GMSC:
  cost-dominates baselines at equal F1. Split-MNIST {0,1}→{2,3}:
  partial EWC forgetting = **-0.030**, naive full forgetting = **+0.386**,
  paired Wilcoxon p = **0.03125** — H3 SUPPORTED on the multi-task benchmark.
  Robustness: 0 % false-alarm on undrifted, GNN AUROC 0.945 → 0.903 under
  50 % edge dropout.
- **Final push (2026-08-30)** — H3 rerun at n=10 with a fair full-with-
  replay baseline: partial forgetting -0.032 vs naive full +0.395 vs
  full+replay -0.017. Wilcoxon **p = 0.0009765625** vs both baselines
  (Wilcoxon minimum at n=10 — partial wins on every single seed).
  Tree adapter at real SLA 0.45: Pareto-position (higher F1 than
  periodic at same actions; 50 % cost saved vs reactive_full).
  Text drift via 1★/5★ → 2★/4★ domain shift: real 4.4 % F1 drop
  (0.969 → 0.924). REST API live (`cadence/api/server.py`), dashboard
  shows real CDAG + real scores, MLflow migrated to sqlite backend.
- **Step G (this section)** — Streamlit dashboard, docker-compose stack,
  auto-generated paper skeleton, weakness register updated.

## Honest open gaps
- **Full Gate A statistical run** (10 seeds × 30 k PPO steps) hasn't landed
  its R-entry yet — the current dashboard falls back to a rule policy.
- **Text drift** (Amazon Reviews / Yelp) not measured — ~7 GB download + text
  classifier scaffolding outside session budget.
- **Live-streaming ingest** is still the batched-window sandbox. The
  Streamlit dashboard is read-only over committed artifacts.
- **Prior-art search** (W-4) for the CDAG novelty claim is not filed.
