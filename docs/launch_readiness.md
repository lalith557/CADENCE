# CADENCE — Launch-Readiness Assessment (2026-09-05)

Honest answer to "is this ready to launch?" **Not as a commercial product yet —
it is a strong research prototype and a compelling demo.** Below is the evidence,
by dimension, with the concrete gap to a launchable MVP.

## Verdict at a glance

| Dimension | State | Launchable? |
|---|---|---|
| Research validity | H1 ✅, H3 ✅ (both p<0.001, n=10); H2 🟡 PRACTICAL (Elec2 69%/1.7% verified), strict-dominance pending Stage 2 | Paper-ready (workshop→) |
| Core system (code) | ~14k lines, all components, 75+ tests, adapters for neural/tree/text | Prototype-complete |
| Runs on real deployments | ❌ everything is sandbox/replay; no integration with a live serving stack | **No** |
| Packaging / install | `pip install -e .`, CLI, resumable runner | Partial |
| Dashboard / demo | Streamlit reads artifacts; CDAG panel was illustrative (W-30) | Demo-grade |
| REST API | FastAPI `/score /monitor /decide /health`, 6/7 tests pass | Prototype |
| Docker / compose | Files exist, **never booted** (daemon offline, W-31) | **No** |
| CI/CD | Workflow written, **never green on a real push** (W-8) | **No** |
| Security / multi-tenancy / auth | Not started (single-node, no authn/z, no tenant isolation) | **No** |
| Observability / SLOs | MLflow experiment tracking only; no product telemetry | **No** |

## What genuinely works (the demo you can show)

- The full closed loop runs end-to-end on Credit Card Fraud in one command:
  drift → PSI trigger → CDAG → GNN+surrogate → RSO → EWC executor → shadow →
  promote/rollback.
- Attribution beats PSI on hard drift (H1), verified at 10-seed scale.
- On real Elec2 drift the policy cuts compute ≈69% for ≈1.7% F1 (H2, verified).
- Partial retraining reduces forgetting on multi-task data (H3).
- Cross-model generality: the same loop drives neural, tree, and text models.
- A pre-registered, reproducible experiment harness with an atomic ledger.

**This is enough to (a) submit a paper and (b) demo to a design partner / investor.**

## The gap to a launchable MVP (ranked)

1. **Prove it on a real, non-sandbox model.** Today all drift is injected or
   replayed and all "production models" are trained in-process. A launch needs at
   least one integration with a real serving stack (a model behind a real
   inference endpoint, real telemetry) — this is the single biggest credibility
   gap for a *product* (it is not a gap for the *paper*).
2. **Complete Stage 2** so the cost-saving claim is confirmatory, not PRACTICAL.
   The sales pitch ("pay a % of GPU-hours we save you") rests on this number.
3. **Boot the stack for real** (Docker/compose), green CI on a push, and a
   clean-clone quickstart that a stranger can run (W-8, W-31).
4. **Productionize the loop**: the RSO runs offline in a sandbox; a product needs
   the online loop wired to a scheduler, with the executor talking to a real
   model registry and rollout controller.
5. **Security & multi-tenancy**: authn/z, tenant isolation, secrets management,
   audit log for retraining decisions (compliance is a selling point in
   fintech/health — currently absent).
6. **Real dashboard** with live CDAG (not illustrative) and per-decision cost
   accounting a customer can act on.
7. **Robustness at scale**: high-dimensional feature spaces, streaming latency,
   partial-telemetry degradation beyond the current small experiments.

## Recommended path

- **Now:** it is *paper-ready* (finish Stage 2 for the strongest H2) and
  *demo-ready* (fix the illustrative dashboard, boot Docker once, capture a
  screencast). Pursue a workshop paper and 1–2 design partners in parallel.
- **3–6 months to a launchable MVP:** items 1, 3, 4, 6 above, scoped to a single
  vertical (e.g. fraud/credit models on one MLOps stack) rather than
  general-purpose. Items 2, 5, 7 harden it toward a first paying customer.

**Bottom line:** the science is real and defensible; the *product* is a strong
prototype that needs real-deployment integration, hardening, and productionization
before it can be sold. Do not market it as production-ready; do demo it and
publish it now.
