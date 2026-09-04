# CADENCE: Causal Attribution-Driven, Cost-Aware Continual Retraining for Production Machine Learning Systems

> Full paper draft. Every quantitative claim traces to a measured `R-` entry in
> `docs/results.md`. The RSO/H2 result is stated as a **controllable cost/quality
> trade-off** (the measured, independently-verified Elec2 result plus a
> pre-registered diagnostic that the policy is healthy); the strict-dominance
> confirmatory run (Stage 2) is ongoing and is scoped here as future work, not
> claimed. Do not strengthen the H2 wording until `R-Gate-A-final` lands.

## Abstract

Every deployed machine-learning model degrades as the data it now sees drifts
away from the data it was trained on. In practice, teams respond either by
retraining on a fixed calendar — wasting large amounts of compute and carbon on
models that have not degraded — or by waiting for a human to notice a metric
drop, by which point real damage has occurred. Crucially, existing production
monitors are *correlational*: they report **that** a feature distribution
shifted, not **why** the model broke or **how much** of it must be repaired. We
present **CADENCE**, a closed-loop system that couples causal drift attribution
to a cost-aware retraining-scope policy. CADENCE builds a *Causal Drift
Attribution Graph* (CDAG) over a unified node set of external input features and
internal model activation clusters, and a graph neural network scores each
node's responsibility for the observed degradation; a counterfactual surrogate,
trained offline on sandboxed interventions, predicts the recovery obtainable from
repairing each node without performing costly ground-truth retrains. These
responsibility scores drive a constrained reinforcement-learning policy — the
*Retraining Scope Optimizer* (RSO) — that chooses among no-op, EWC-regularized
partial retraining, and full retraining to minimize compute and carbon cost
subject to a service-level-agreement (SLA) recovery constraint; the chosen fix is
validated in shadow mode before promotion. Across neural-network, tree-ensemble,
and text production models we find that (i) the learned attribution localizes the
true injected root cause significantly more precisely than PSI/KS-test baselines
on hard concept-shift and gradual-drift scenarios (mean-reciprocal-rank and AUROC
gains, *p* < 0.001 over ten seeds); (ii) on a real concept-drift benchmark the
policy delivers a controllable cost/quality trade-off, cutting retraining compute
by ≈69% for a 1.7-point F1 reduction versus a reactive full-retrain baseline;
and (iii) causally-targeted partial retraining reduces catastrophic forgetting on
unrelated tasks relative to naive full retraining (*p* < 0.001). We further report
a pre-registered diagnostic study of the RSO's learning dynamics that surfaces —
and repairs — a reward-scaling pathology, and we release a reproducible harness
with an atomic experiment ledger and pre-registered success criteria. CADENCE
reframes model maintenance from a fixed schedule into a self-healing loop that
acts only when, where, and as much as the evidence warrants.

---

## 1. Introduction

*(As in `front_matter.md` §1 — six paragraphs: the two unsatisfying operational
responses to drift; the correlational-vs-causal gap; CADENCE's closed-loop
architecture; the two design choices — narrow adapter interface, and
synthetic-for-attribution / real-for-outcome validation; a results preview; and
the three-part contributions list. Reproduced verbatim in the assembled draft.)*

**Contributions.** **(1)** A unified, online causal attribution mechanism (the
CDAG plus a jointly-trained GNN and counterfactual surrogate) that reaches into
model internals and identifies the root cause of production degradation more
precisely than correlational detectors. **(2)** A formulation of retraining
*scope* selection as a compute- and carbon-aware constrained RL problem, and
evidence — from a real-drift benchmark and a pre-registered learning-dynamics
diagnostic — that a single learned policy provides an operator-tunable
cost/quality trade-off unavailable to fixed rules. **(3)** A model-agnostic,
forgetting-safe execution loop with an experimental protocol that validates
attribution on synthetic ground-truth and end-to-end value on real drift, plus a
reproducible harness (atomic ledger, pre-registered gates, released code).

---

## 2. Related Work

CADENCE draws from four communities; see also `docs/prior_art.md` for the
per-claim prior-art search and the narrowing of each novelty claim.

**Statistical drift detection.** Population Stability Index and KS/Hotelling
monitors, productised by Evidently, WhyLabs, Arize, and Fiddler and rooted in the
concept-drift literature (ADWIN and the Bifet–Gama family), report *that* an
input distribution moved. They are correlational and cannot say whether a shift
is causally responsible for a performance drop, nor which internal component it
broke — the gap CADENCE's attribution targets. We use PSI both as CADENCE's own
trigger (the "smoke alarm") and as the attribution baseline to beat.

**Causal discovery for model debugging.** Constraint-based (PC) and
continuous-optimization (NOTEARS) structure learning, and recent drift-specific
extensions ("Causal Explanation of Concept Drift", TempXAI 2025; "Tracing
Distribution Shifts with Causal System Maps", 2025), build causal graphs over
*feature space*. CADENCE's distinguishing construction is a single graph whose
nodes span **both** external features and **internal activation clusters**,
updated online — a construction the surveyed feature-space work does not occupy.

**Continual learning.** EWC, LwF, SI, and replay methods protect against
catastrophic forgetting *during* retraining but assume the decision to retrain
has already been made; they optimize *how* to retrain, not *whether* or *how
much*. CADENCE consumes EWC as its partial-retrain executor and asks the prior
question.

**Constrained RL and Green AI.** Our RSO is an application of constrained MDP
methods (Constrained Policy Optimization; Lyapunov/augmented-Lagrangian safe RL)
with a reward that adds a carbon term motivated by the Green-AI literature. The
novelty is not the CMDP algorithm but the *object*: framing retraining *scope*
over `{no-op, partial-subnetwork, full}` for a deployed model as the action
space, with cost + carbon + SLA in one reward. A prior-art search found MLOps
"when to retrain" (timing) work but not retraining *scope* as a learned policy.

---

## 3. System Design

Figure 1 shows the closed loop. CADENCE reaches the production model only through
a narrow **adapter interface** — `score`, `tap_activations`, `partial_fit`,
`full_fit` — so the identical loop drives a feed-forward neural network, a
gradient-boosted-tree ensemble, and a linear text classifier; a model class that
cannot support targeted partial retraining (e.g. a tree) raises
`NotImplementedError`, which the executor catches and escalates to a full retrain.

**3.1 Monitoring and triggering.** A streaming collector samples PII-hashed
feature vectors, the model's predictions, delayed ground-truth labels, and
internal activations at chosen tap points. A PSI + delayed-label-F1 trigger fires
when a rolling window crosses threshold — deliberately the *same* correlational
mechanism CADENCE's attribution then out-performs, so the comparison is fair: PSI
does the alerting; CADENCE's value-add is the *attribution*.

**3.2 The Causal Drift Attribution Graph.** On an alert, CADENCE builds a CDAG
over three node types: one per input feature's windowed statistics; one per
internal activation cluster (k-means over each tapped layer's neuron activations);
and one performance node. A hybrid procedure fits the graph — PC constraint tests
decide which edges may exist (the skeleton), then a NOTEARS-style continuous
relaxation assigns real-valued weights restricted to that skeleton.

**3.3 GNN attribution and the counterfactual surrogate.** A GraphSAGE/GCN-style
GNN embeds each node in the context of its causal neighborhood. A counterfactual
surrogate — an MLP head trained jointly with the GNN — maps each node embedding
plus drift context to the model recovery that repairing that node would yield.
The surrogate's labels come *offline* from a sandbox: inject a known drift,
actually retrain to fix it, measure the resulting F1, and record the
`(node, intervention, post-fix F1)` triple. At inference the surrogate does a
forward pass — a **responsibility score** — so no ground-truth intervention is
needed on the live model. Rolled-back deployments feed their realized outcomes
back as new triples, closing a calibration loop.

**3.4 The Retraining Scope Optimizer (RSO).** The responsibility scores, current
F1, SLA margin, attribution concentration (a Herfindahl index over the scores),
per-action compute/carbon estimates, and time-since-retrain form the state of a
PPO policy over `{no-op, partial, full}`. The reward is

  `r = Δf1 − w · cost_scale · gpu_cost − λ · max(0, SLA − post_f1)`,

with the SLA constraint enforced by an augmented-Lagrangian dual variable λ
updated between episodes by projected ascent on the (smoothed) violation. A
`cost_scale` constant lifts the otherwise-negligible compute term (FraudNet is
~4k FLOPs/forward) into a regime the policy can perceive.

**3.5 Execution and validation.** On `partial`, the executor freezes all but the
implicated subnetwork, computes Fisher information on a cached sample of the
original data, and fine-tunes under an EWC penalty with a replayed mix of
historical and recent data. On `full`, it retrains the whole model. Either
candidate is scored in **shadow** on a held-out recent window; it is promoted
only if its F1 clears the SLA, else rolled back — and the (predicted, realized)
pair recalibrates the surrogate. A diffuse-responsibility guard escalates to full
retraining when no single node dominates; an unrecoverable-drift guard stops
burning compute when repeated retrains fail to recover (e.g. a label-rule flip).

---

## 4. Experimental Setup

**Datasets.** Credit Card Fraud (neural adapter; 284k transactions, 0.17%
positive), Give Me Some Credit (tree adapter), a TF-IDF/LogReg text adapter on
Yelp reviews, the Electricity (Elec2) and Airlines streaming benchmarks for real
drift, and Split-MNIST for multi-task forgetting.

**The two-track validation principle.** Attribution is validated where an answer
key exists — **synthetic** drift whose injected root cause is known — reporting
root-cause precision/recall/AUROC. End-to-end value is validated on **real** drift
where the cause is unknown, graded by *outcome* (recovery and cost). This lets us
make a precise attribution claim without overclaiming on data that has no ground
truth.

**Baselines.** For attribution: PSI (correlational) and a CDAG-structural
path-sum proxy (no learned readout). For the system: reactive-full-retrain (the
common production reflex), fixed-period retraining, and no-op.

**Protocol.** ≥10 seeds for the headline attribution and forgetting results;
paired Wilcoxon signed-rank with Bonferroni correction; bootstrap 95% CIs. All
runs are tracked in MLflow; the RSO study uses a pre-registered gate battery
(§6) and an atomic, resumable experiment ledger.

---

## 5. Results

### 5.1 H1 — Causal attribution beats correlational detection (Figure 2)

On the synthetic benchmark (5 scenarios × 10 seeds, `R-Gate-B-n10-subset`), we
split scenarios by whether PSI is *saturated*. On the **hard** subset —
`time_gradual` and `v14_concept_shift`, where PSI's top-1 accuracy is 0 — the
learned CDAG+GNN scorer beats PSI on mean-reciprocal-rank (**0.412 vs 0.097**)
and AUROC (**0.905 vs 0.672**), paired Wilcoxon **p < 0.001** on both (n = 20
paired), and beats the structural proxy at p = 0.032. On the **easy** abrupt
subset PSI is already saturated at 1.0, and the GNN is not-inferior (AUROC 0.989).
The all-scenarios MRR "tie" (p = 0.28) is a mixing artefact of averaging over
PSI's saturated ceiling; the AUROC test survives it (p < 0.001). **H1 is
supported on the contested subset at paper scale**, and the structural proxy
losing to PSI shows the learned readout — not merely the causal graph — does the
work. A graph-recovery caveat (SHD-proxy = 1.0 on short windows) is noted as a
limitation: the GNN wins by message-passing over soft signal, not by recovering a
clean edge, which we discuss in §7.

### 5.2 H2 — A controllable, cost-aware retraining policy (Figure 3)

On the **Elec2** real-drift benchmark (`R-Gate-E-verified`, independently
recomputed from raw per-seed logs), a temporal 30/70 train/stream split produces
a genuine 41.5-point undrifted-F1 drop. Against the reactive-full-retrain
baseline, CADENCE **saves 69.1% of retraining compute for a 1.66-point F1 cost**
(95% bootstrap CI on the F1 gap [−0.025, −0.009]); against periodic retraining it
wins +0.16 F1 at 1.54× the cost. Neither baseline dominates CADENCE and CADENCE
dominates neither — it is a **Pareto-frontier option** exposing an operator knob
(the reward weights) that fixed rules lack. Under our pre-registration this is a
**PRACTICAL (Claim-B)** result: a real, quantified cost/quality trade-off. The
strict-dominance (Claim-A) question — whether a fully-trained policy strictly
beats every baseline on both axes — is the confirmatory Stage-2 experiment, still
in progress (§6, §8); we do not claim it here.

### 5.3 H3 — Causally-targeted partial retraining reduces forgetting (Figure 4)

On Split-MNIST (`R-Gate-F-mnist-n10`, tasks {0,1} then {2,3}, 10 seeds),
EWC-regularized partial retraining of the implicated layer preserves the first
task almost perfectly (forgetting −0.032, a slight *gain*) while naive full
retraining loses **0.395 F1** on it. Paired Wilcoxon partial < full: **p < 0.001**;
partial also beats the stronger full-with-replay baseline. On single-task Credit
Card Fraud the effect vanishes (`R-Gate-D`, an honest negative), which together
give the correct claim: **EWC's forgetting protection materialises when there is a
distinct task to preserve**; on single-task deployments a full-retrain-with-replay
is already sufficient.

### 5.4 Generality and ablations

The identical loop runs over the LightGBM tree adapter (Give Me Some Credit) and
the TF-IDF/LogReg text adapter (Yelp), with the `NotImplementedError → full
retrain` contract engaging correctly for the tree. The partial ablation table
(`docs/paper/ablations_partial.md`) shows both **no-GNN** (structural readout
only) and **no-surrogate** (PSI only) cause large, significant drops on the hard
subset (MRR Δ +0.26 / +0.32, p ≤ 0.032) — neither the causal graph nor the
learned readout is decorative. The `no-PPO`/`no-contested`/`no-augmented-Lagrangian`
ablations are gated on Stage 2.

---

## 6. A Pre-Registered Diagnostic of the RSO's Learning Dynamics

A distinguishing methodological contribution is that we treated the RSO's
training as an object of study rather than a black box. Before the confirmatory
run we **pre-registered** eight sanity gates and four outcome verdicts
(`docs/gate_a_preregistration.md`) and drove the policy through a single-fix
diagnostic loop. This surfaced and repaired a real pathology: at the naive reward
scaling the compute term was ~10⁹× smaller than the F1 term (audit §1.1), so the
policy was blind to cost and collapsed to bang-bang; rescaling restored a
cost-responsive policy. A second pathology — the augmented-Lagrangian dual
variable saturating at its cap because the contested-scenario mix contained a
*permanently unrecoverable* scenario — was diagnosed from the measured violation
distribution and fixed by tolerating the irreducible violation floor. Two
originally pre-registered gates were then shown, with evidence, to be
*mis-specified* (a cost-visibility ratio provably in tension with a
non-collapsing policy; an F1-variance gate that penalized reproducible seeds) and
re-specified via a signed amendment. The resulting policy passes the
policy-health gates (no action collapse; healthy exploration entropy; both cost
and SLA penalties shrink over training; a stable dual variable). We report this
loop in full because the honest exposure of reward mis-scaling and gate
mis-specification is, we argue, exactly the rigor ML-systems evaluation usually
lacks — and it is reproducible from the released ledger.

---

## 7. Limitations and Threats to Validity

- **H2 strict dominance is not claimed.** The confirmatory 10-seed × 30k-step
  Stage-2 run is ongoing; the paper's H2 claim is the verified PRACTICAL
  trade-off, not Pareto dominance.
- **Graph recovery is weak on short windows** (SHD-proxy = 1.0); attribution
  succeeds via GNN message-passing rather than clean edge recovery. Interpretable
  causal graphs would need longer windows / looser sparsity.
- **Concept-flip drift is unrecoverable** by any retrain without new labels — a
  fundamental limit we detect and escalate rather than paper over.
- **Illustrative design choices** (FraudNet MLP, k-means clustering, fixed tap
  points) are labelled as method choices, not discoveries.
- **Single-node H3 negative** and small-n caveats on some ablations are reported
  rather than hidden.

---

## 8. Conclusion and Future Work

CADENCE couples causal, model-internals-aware drift attribution to a cost-aware
retraining-scope policy and a forgetting-safe executor, and shows — with honest,
reproducible evidence — that attribution can beat correlational monitoring on hard
drift (H1), that a learned policy exposes a real cost/quality knob on live drift
(H2, PRACTICAL), and that causally-targeted partial retraining reduces forgetting
where a task exists to preserve (H3). Future work: complete the Stage-2
confirmatory run to test strict dominance; extend the CDAG to multimodal and LLM
drift; and give the surrogate approximation-error guarantees. The system, harness,
pre-registration, and ledger are released for reproduction.

---

*Figures: Fig 1 `fig_architecture`; Fig 2 `fig_h1_subset`; Fig 3
`fig_elec2_pareto`; Fig 4 `fig_h3_mnist_forget`. Captions in
`docs/paper/figures/README.md`. All numbers trace to `docs/results.md`.*
