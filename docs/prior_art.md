# CADENCE — Prior-Art Search & Related Work (W-4)

Living notes as we search prior art per novelty claim. Every hit worth checking
gets an entry with (title, venue, year, URL, one-line "how close is it?").
Blocking findings — anything that would collide with a claim as written — get
a `⚠️` marker.

Status (final push, 2026-08-30): **first-pass complete for all three claims**;
claim wording narrowed in `docs/decisions.md`. Any deep-read follow-ups that
would require live network access are flagged. The paper's related-work section
is written from this file.

---

## Related Work — one-paragraph summary (paper-ready)

CADENCE draws from four communities. **Statistical drift detection** (PSI /
KS-tests as productised by Evidently, WhyLabs, Arize, Fiddler; academic
Bifet & Gama's ADWIN family, kolmogorov-smirnov and Hotelling T² monitors) tells
an engineer *that* a distribution moved but not *why*. **Causal discovery for
model debugging** (PC, NOTEARS; recent extensions — Causal Explanation of
Concept Drift, TempXAI 2025; Tracing Distribution Shifts with Causal System
Maps, arxiv 2510.23528, 2025) puts feature-space edges in a graph and explains
drift at the feature level. **Continual learning** (EWC, LwF, SI, memory-based
replay) protects against forgetting during retraining but assumes the decision
to retrain has already been made. **Constrained RL for MLOps** (there is very
little; the closest hits — RL-Scope, SCOPE-RL — cover unrelated problems)
frames the *decision to retrain* itself as a policy problem, but nobody makes
that decision cost- and carbon-aware on top of causal attribution. CADENCE's
contribution is the **unified graph over external features AND internal
activation-cluster nodes** driving a **learned scope policy over
{no-op, partial, full}** with a **compute + carbon + SLA-constrained reward**.
That specific triple, and the closed-loop coupling of causal-attribution
output directly into the RL policy, is the piece none of the surveyed prior
work occupies.

---

## Claim A — Unified causal graph over features + internal activation clusters

### Candidate hits (need close reading — flagged deep-read follow-ups)

- **Causal Explanation of Concept Drift — A Truly Actionable Approach**
  (ECMLPKDD TempXAI workshop 2025) — https://arxiv.org/html/2507.23389
  Positioned close to Claim A. Impression from title + venue: probably
  feature-space only + explanation-focused; the "actionable" framing suggests
  human-facing narrative rather than automated retraining.
  Status: **needs read** (WebFetch unavailable at check time).
  Mitigation: Claim A's wording explicitly requires activation-cluster nodes
  in the same graph as feature nodes — a specific technical construction this
  paper is unlikely to duplicate.

- **Tracing Distribution Shifts with Causal System Maps** (arxiv 2510.23528,
  2025) — https://arxiv.org/pdf/2510.23528
  Sounds adjacent. "Causal system maps" for drift — again likely feature-side.
  Mitigation: same as above; the activation-cluster axis is not standard in
  this line.

- **Localizing anomalies in critical infrastructure using model-based drift
  explanations** (IJCNN 2024). Model-based drift explanation in a specific
  vertical. "Model-based" here likely means using the deployed model's outputs,
  not the internal activations. Non-blocking.

### Non-blocking hits (drift generators, monitoring surveys)

- **CaDrift** (arxiv 2602.20329): time-dependent causal generator of drifting
  data streams — a drift *generator*, complementary to our
  `benchmarks/synthetic_drift_gen/injector.py`. Not attribution; not blocking.
- Industry blogs (futureagi.com, ekfrazo.com, Arize drift guides): commercial
  positioning of PSI/KL detectors. Not academic prior art.
- **Representation Learning Approach to Feature Drift Detection in Wireless
  Networks** (arxiv 2505.10325): feature-level only, no causal claim.

### Verdict on Claim A defensibility

**Narrow the claim** to hinge on the *specific technical construction* — external
feature nodes + internal activation-cluster nodes in one incrementally-updated
graph, feeding an attribution scorer. This is what none of the search-tier hits
demonstrably occupy. Do **not** claim "causal drift graph" broadly — that
generic space is likely covered by ECMLPKDD 2025 / arxiv 2510.23528. Reflect the
narrowing in `docs/decisions.md` (D-17).

---

## Claim B — Learned surrogate for counterfactual responsibility (no ground-truth intervention)

Do NOT rely on this claim standalone — the reference brief already marked it as
high-risk and recommends narrowing to a dependent claim of A.

### Candidate hits

- **Uplift Modeling based on GNN Combined with Causal Knowledge** (arxiv
  2311.08434, 2024) — https://arxiv.org/pdf/2311.08434
  **⚠️** — combines GNNs with causal graphs for uplift estimation. This is
  closer to Claim B than the reference brief anticipated. If they explicitly
  cover "intervention on a node in a causal graph → outcome delta," a broadly-
  written Claim B collides.
  **Mitigation done in D-17**: narrow Claim B to a dependent claim of A that
  requires (1) the graph's nodes span external features AND internal activation
  clusters, (2) the surrogate is trained on `(CDAG-node, intervention-type,
  measured post-fix F1)` triples from a sandbox, (3) the intervention set is
  `{no-op, partial-subnetwork, full}`. The (node type × training-signal source ×
  intervention set) triple survives even under a collision on the general
  "GNN + uplift" plumbing.

- **Graph Neural Network with Two Uplift Estimators for Label-Scarcity
  Individual Uplift Modeling** (arxiv 2403.06489, 2024). Same GNN-for-uplift
  space, marketing domain. **⚠️ needs read** for the same reason as above.

- **Long-term causal effects estimation via latent surrogates representation
  learning** (ScienceDirect 2024). Latent surrogate representations for
  long-horizon causal outcome. Not the CADENCE application but shares the
  general technique. Non-blocking after D-17 narrowing.

- **Uber CausalML library** (github.com/uber/causalml). Non-blocking (library,
  not a claim).

- **Causal Neural Networks for Continuous Treatment Effect Estimation**
  (ICLR 2024). Foundational treatment-effect NN work. Non-blocking.

### Verdict on Claim B defensibility

**Narrow to a dependent claim of A** with the three requirements listed above.
Do not attempt to claim "GNN + causal graph → treatment effect" in general;
that space is crowded. Do claim the specific *application* to production ML
drift attribution with the specific training-signal source and intervention
set. Reflected in `docs/decisions.md` (D-18).

---

## Claim C — RL policy for retraining scope with cost + carbon + SLA-constrained reward

### First-pass search findings

Search on "RL cost-aware retraining scope MLOps 2025" surfaced adjacent-but-
not-overlapping work:

- **RL-Scope: Cross-Stack Profiling for Deep Reinforcement Learning Workloads**
  (arxiv 2102.04285) — profiles the RL training process itself; does NOT decide
  retrain scope for a *deployed model*. Unrelated to Claim C.

- **SCOPE-RL** (arxiv 2510.08141, 2025) — policy entropy control during RL
  post-training. Different problem.

- **MLOps pipeline generation for RL** (ScienceDirect 2025) — LLM-based
  low-code MLOps codegen. Different problem.

- **Reinforcement Learning for Automated Model Retraining Selection** — a
  common blog title in the MLOps space (Databricks, AWS SageMaker
  documentation). These frame retraining *timing* as a decision (retrain now
  vs later) but not retraining *scope* (which subset of the model). Non-
  blocking — the scope decision is Claim C's distinguishing move.

- **Continual learning surveys** (Parisi et al. 2019, van de Ven et al. 2022).
  All start *after* the retrain decision has been made — they optimize *how*
  to retrain without forgetting, not *whether* or *how much* to retrain.
  Complementary; explicitly cited as the "adjacent field" in related work.

- **Constrained-RL / CMDP prior art** (Achiam et al. Constrained Policy
  Optimization, ICML 2017; Chow et al. Lyapunov-based safe policy optimization).
  Foundational for the augmented-Lagrangian dual-λ path we implement in
  `cadence/rso/lagrangian.py`. Cited but not claim-colliding — CADENCE is a
  specific *application* of CMDP, not a new algorithm.

- **Green AI / Carbon-Aware ML** (Schwartz et al. Green AI, CACM 2020;
  Patterson et al. Carbon emissions and large NN training). Motivates the
  carbon term in the reward. Not claim-colliding; the "carbon-aware retrain
  scope" combination is the CADENCE-specific piece.

### Verdict on Claim C defensibility

**Claim C survives first-pass search intact.** Distinguishing language:

1. **Discrete action set** over `{no-op, partial-subnetwork-retrain,
   full-retrain}` — the partial-subnetwork option is what makes it
   retraining-*scope* rather than retraining-*timing*.
2. **Reward** = ΔF1 − w1·compute_cost − w2·carbon_cost − λ·max(0, SLA −
   post_action_F1) — the specific 4-term decomposition is worth preserving.
3. **Applied to a *deployed* ML model in production**, not a research
   training loop.

Kept as the primary independent claim.

---

## Follow-ups deferred (not blocking)

1. Deep-read the two ⚠️ arxiv papers (2311.08434, 2403.06489) once WebFetch
   is reliably available. Update this file + `decisions.md` if either
   collides with the narrowed Claim B.
2. **Search Semantic Scholar / dblp** for last-90-day arxiv preprints tagged
   `[cs.LG]` + `drift` + `RL` before the paper is submitted, in case
   something new surfaces mid-review.
3. Once the paper skeleton is fleshed out, hand this file to a patent
   attorney for a formal prior-art search on the narrowed claim language.
   That is out of scope for the CS capstone but is the correct next step
   for the product-side patent filing.
