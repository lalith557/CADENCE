# CADENCE — Prior-Art Search Notes (W-4)

Living notes as we search prior art per novelty claim. Every hit worth checking gets an entry with (title, venue, year, URL, one-line "how close is it?"). Blocking findings — anything that would collide with a claim as written — get a `⚠️` marker.

Status: **partial**. First pass done for Claim A on 2026-08-28; RL / surrogate searches pending due to transient tool availability.

---

## Claim A — Unified causal graph over features + internal activation clusters

### Candidate hits (need close reading)

- **Causal Explanation of Concept Drift — A Truly Actionable Approach** (ECMLPKDD TempXAI workshop 2025) — https://arxiv.org/html/2507.23389
  Positioned close to Claim A. Need to read to determine: (a) does it use activation clusters as graph nodes, or only external features? (b) does it drive automated retraining, or only human-facing explanation? Impression from title + venue: probably feature-space only + explanation-focused, but must verify.
  Status: **needs read** (WebFetch unavailable at check time).

- **Tracing Distribution Shifts with Causal System Maps** (arxiv 2510.23528, 2025) — https://arxiv.org/pdf/2510.23528
  Sounds directly adjacent. "Causal system maps" for drift — again need to check activation coverage + retraining loop.
  Status: **needs read** (WebFetch unavailable at check time).

- **Localizing anomalies in critical infrastructure using model-based drift explanations** (IJCNN 2024)
  Model-based drift explanation in a specific vertical. Likely closer to industrial anomaly detection than to model-internal attribution, but worth checking whether "model-based" means what CADENCE means.
  Status: **needs read**.

### Non-blocking hits (drift generators, monitoring surveys)

- CaDrift: time-dependent causal generator of drifting data streams (arxiv 2602.20329) — **drift generator**, not attribution. Complementary; could be an additional benchmark alongside `benchmarks/synthetic_drift_gen`.
- futureagi.com / ekfrazo.com model-drift guides — industry blog posts, not academic prior art.
- "Representation Learning Approach to Feature Drift Detection in Wireless Networks" (arxiv 2505.10325) — feature-level, no causal claim.

### Preliminary read on Claim A's defensibility

None of the search-tier hits explicitly claim to bind *activation-cluster nodes* into the same causal graph as external features, then drive an *automated retraining scope decision* from that graph. That is CADENCE's specific construction and is the piece Claim A rests on. But:
- **Rewrite the claim to hinge on "external feature nodes + internal activation-cluster nodes in a single incrementally-updated graph"** — this is the specific technical piece the search hasn't collided with.
- Do not claim "causal drift graph" broadly — the ECMLPKDD 2025 and arxiv 2510.23528 papers likely occupy that generic space.

---

## Claim B — Learned surrogate for counterfactual responsibility (no ground-truth intervention)

Do NOT rely on this claim standalone — the reference brief already marked it as high-risk and recommends narrowing to a dependent claim of A.

### Candidate hits from first pass

- **Uplift Modeling based on GNN Combined with Causal Knowledge** (arxiv 2311.08434, 2024) — https://arxiv.org/pdf/2311.08434
  Combines GNNs with causal graphs for uplift estimation. This is CLOSER TO CLAIM B than expected — need to determine whether it targets *drift-attribution intervention* (our specific application) or *marketing uplift* (their standard application). If they explicitly cover "intervention on a node in a causal graph → outcome delta," Claim B collides. **⚠️ needs read.**
- **Graph Neural Network with Two Uplift Estimators for Label-Scarcity Individual Uplift Modeling** (arxiv 2403.06489, 2024) — same GNN-for-uplift space, likely marketing-domain, but the "graph node → treatment effect" plumbing is shared. **⚠️ needs read.**
- **Long-term causal effects estimation via latent surrogates representation learning** (ScienceDirect 2024) — surrogate representations for long-horizon causal outcome — again the closest technical language.
- **Uber CausalML library** (uber/causalml on GitHub) — full library for uplift + meta-learners. Non-blocking (a library, not a claim).
- **Causal Neural Networks for Continuous Treatment Effect Estimation** (ICLR 2024) — general treatment-effect NN; foundational.

### Preliminary read on Claim B's defensibility

The uplift-modeling-on-GNN space is more crowded than the reference brief anticipated. Two arxiv papers already do "GNN + causal graph → treatment-effect estimate." Even if their application is marketing rather than MLOps drift, the *technical construction* is very close to Claim B.

**Recommendation:** narrow Claim B to a dependent claim of Claim A that explicitly requires:
1. The graph's nodes span external features AND internal activation clusters (from A).
2. The surrogate is trained specifically on `(CDAG-node, intervention-type, MEASURED post-fix F1)` triples generated by running retrains in a sandbox.
3. The intervention set is `{no-op, partial-subnetwork-retrain, full-retrain}` (from C).

That combination of (node type × training-signal source × intervention set) is what would still be defensible after a full patent-attorney review. Broader Claim B language will likely collide.

---

## Claim C — RL policy for retraining scope with cost + carbon + SLA-constrained reward

### First-pass search findings

Search on "RL cost-aware retraining scope MLOps 2025" surfaced adjacent-but-not-overlapping work:
- **RL-Scope: Cross-Stack Profiling for Deep Reinforcement Learning Workloads** (arxiv 2102.04285) — profiles RL training, does NOT decide retrain scope for a *deployed model*. Unrelated to Claim C.
- **SCOPE-RL** (arxiv 2510.08141, 2025) — policy entropy control during RL post-training. Different problem.
- **MLOps pipeline generation for RL** (ScienceDirect 2025) — LLM-based low-code MLOps codegen. Different problem.

No hit collides with Claim C's specific formulation of "learned policy over {no-op, partial, full} retrain scope for a deployed ML model with a compute+carbon+SLA-constrained reward."

**RLOps as a term exists** and covers "MLOps applied to RL policies," but that's about deploying an RL policy — the inverse of using RL to decide retrain scope for a supervised model. Terminology confusion is worth flagging in the related-work section but not a claim collision.

### Preliminary verdict on Claim C's defensibility

Strongest claim per the reference brief holds up under first-pass search. Distinguishing language for the claim:

1. **Discrete action set** over `{no-op, partial-subnetwork-retrain, full-retrain}` — the partial-subnetwork option is what makes it retraining-*scope* rather than retraining-timing.
2. **Reward** = ΔF1 − w1·compute_cost − w2·carbon_cost − λ·max(0, SLA − post_action_F1) — the specific 4-term decomposition is worth preserving in the claim.
3. **Applied to a *deployed* ML model in production**, not a research training loop.

Kept as the primary independent claim.

---

## Next steps

1. Retry WebFetch on the two named candidate URLs when the classifier is back up.
2. Do the missing Claim B + Claim C searches.
3. If any candidate lands within one-hop of Claim A as written, revise the claim language BEFORE the paper skeleton pulls it in.
4. Timebox the whole prior-art pass at 90 minutes total. Return to CDAG/PPO/product engineering after that.
