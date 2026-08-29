# CADENCE — Paper Skeleton

Target venues (in priority order): **IEEE Big Data 2027**, **ACM SoCC**, **EuroMLSys**, **NeurIPS ML for Systems workshop** as a fallback.

## Working title
> **CADENCE**: Causal Attribution-Driven Efficient Continual Retraining for Production ML Systems

## Abstract (target 200 words)
_TODO — will be auto-filled from `docs/results.md` headline numbers once H1/H2/H3 land._

Elements to include, in order:
1. Problem: production ML degrades; today's tools detect drift statistically but not causally, and retraining decisions are rules-of-thumb.
2. Contribution: CDAG (unified feature+activation causal graph) + counterfactual surrogate + PPO-based cost/carbon/SLA-aware retraining scope optimizer.
3. Method: hybrid PC+NOTEARS causal discovery; GNN embedding; surrogate trained on sandbox interventions; EWC-regularized partial retrain.
4. Results (blanks to fill in): attribution precision Δ vs PSI baseline; compute reduction vs periodic-retrain baseline; forgetting reduction vs naive full retrain.
5. Availability: open-source, Docker-reproducible.

## Section outline
1. **Introduction** (1 page) — problem framing, three hypotheses (H1/H2/H3), contributions list.
2. **Related work** (0.75 page) — drift detection (Evidently, WhyLabs, SageMaker Monitor, Fiddler); continual learning (EWC, replay, LwF); causal discovery (PC, NOTEARS); RL for MLOps.
3. **Method** (2.5 pages)
   3.1. CDAG construction — hybrid PC+NOTEARS + activation-cluster nodes (Claim A)
   3.2. Counterfactual attribution via learned surrogate (Claim B — narrow)
   3.3. Retraining Scope Optimizer — PPO with Lagrangian SLA constraint (Claim C)
   3.4. EWC-regularized executor + shadow/canary
4. **Experimental setup** (0.75 page) — datasets, drift injectors, baselines, ablations, hardware, seeds, MLflow reproducibility appendix pointer.
5. **Results** (2 pages, all tables auto-populated from `docs/results.md`)
   5.1. H1 — attribution precision vs correlational baselines (Fraud + synthetic drift ground truth)
   5.2. H2 — compute & carbon savings vs periodic + reactive baselines (RSO)
   5.3. H3 — forgetting mitigation vs naive full retrain (Split-MNIST via Avalanche)
   5.4. Generality — tree-ensemble adapter results on Give Me Some Credit
   5.5. Ablations — no-GNN, no-surrogate, rule-based scope
6. **Discussion** (0.5 page) — failure modes, threats to validity, when CADENCE hurts.
7. **Limitations** (0.25 page) — explicit list.
8. **Conclusion + future work** (0.25 page).
9. **Reproducibility appendix** — Docker image, config hashes for every table, seed list, hardware profile.

## Figures budget
- Fig 1: Architecture diagram (from `CADENCE-master-reference.md` §6).
- Fig 2: CDAG example on FraudNet (auto-generated).
- Fig 3: H1 precision/recall curve.
- Fig 4: H2 cost vs SLA recovery scatter.
- Fig 5: H3 forgetting curve.
- Fig 6: Ablation bar chart.

## What's not in the paper (kept for a v2)
- Full LLM behavioral drift extension.
- Federated CDAG.
- Multimodal.

## Reviewer objections we already know to address
- "Why not just use SHAP for attribution?" — SHAP is per-prediction, not per-drift-episode; also correlational.
- "Why PPO not a bandit?" — bandit doesn't handle the sequential-decision nature of shadow→promote→revisit.
- "Isn't causal discovery unreliable on small windows?" — hybrid PC+NOTEARS + our windowing choice + validation via SHD on synthetic data addresses this; see §5.1.
- "How do you avoid selection bias in the surrogate training data?" — sandbox injections cover both true-cause and confounder scenarios uniformly, not just historical picks.
