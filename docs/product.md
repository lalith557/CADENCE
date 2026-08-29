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
