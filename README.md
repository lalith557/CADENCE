# CADENCE: Causal Attribution-Driven Efficient Continual Retraining for Production ML Systems

---

## SECTION 1: Professional Project Title

**CADENCE** — *Causal Attribution-Driven Efficient Continual Retraining for Production ML Systems*
(alt. name for GitHub/branding: **Sentinel-C** or **CausalOps**)

---

## SECTION 2: The Real-World Problem

Every deployed ML model degrades over time as the world it was trained on drifts away from the world it now sees — pricing models, fraud detectors, recommendation rankers, medical risk scorers, LLM-based agents. Industry surveys consistently report that model performance decay from data/concept drift is one of the top causes of production ML failure, and that most teams either (a) retrain on a fixed calendar schedule regardless of need, wasting enormous compute and carbon, or (b) wait for a human to notice a KPI drop, by which point real damage (financial, medical, reputational) has already occurred. Retraining a large production model can cost anywhere from thousands to millions of dollars in compute, so the *decision of when, what, and how much to retrain* is itself a high-stakes, poorly-automated problem. No widely deployed system today can tell an engineer **why** a model broke — which specific upstream causal factor is responsible — only **that** something drifted, statistically.

---

## SECTION 3: State of the Art and Its Limitations

- **Statistical drift detectors** (Evidently AI, WhyLabs, Amazon SageMaker Model Monitor, Fiddler): use KS-tests, Population Stability Index, or Jensen-Shannon divergence on feature distributions. These are purely **correlational** — they flag that a distribution shifted, not whether that shift is *causally responsible* for the performance drop, leading to high false-alarm rates and wasted retrains.
- **Full periodic retraining pipelines** (most MLOps stacks, e.g. weekly/monthly cron retrains): computationally wasteful, ignore whether drift is even relevant to the deployed task, and have a large carbon/cost footprint at scale.
- **Continual learning / online learning methods** (EWC, replay buffers, LwF): address *catastrophic forgetting* during retraining but say nothing about *when* to trigger retraining or *which sub-components* of the model actually need updating.
- **Causal discovery literature** (PC algorithm, NOTEARS, causal graphical models): mostly evaluated on small, static, offline datasets — not integrated into a live production monitoring-and-retraining loop, and not combined with model internals (which layers/features/subnetworks a drift actually touches).

**Gap**: nobody has combined (1) online causal attribution of *why* a production model degraded, with (2) a resource-aware decision policy for the *minimal sufficient retraining scope* (none / partial-subnetwork / full), producing a closed-loop, cost- and carbon-aware self-healing ML system.

---

## SECTION 4: Proposed AI/ML Innovation

**CADENCE** is a closed-loop MLOps system with a novel core algorithm, the **Causal Drift Attribution Graph (CDAG)**:

1. It continuously builds and updates a lightweight causal graph over (a) upstream data-generating features, (b) internal model activation clusters (via GNN-based embedding of layer/neuron groups), and (c) downstream performance metrics.
2. When performance degrades, CADENCE runs **counterfactual intervention scoring**: it asks, for each upstream feature/subsystem, "if this factor's distribution were reverted to its training-time distribution, would the performance drop disappear?" — estimated via a fast surrogate using proxy interventions, not full retraining.
3. This produces a **ranked causal responsibility score** per feature/module, which feeds a **Retraining Scope Optimizer (RSO)**: a constrained RL policy that chooses among {no-op, targeted subnetwork fine-tune, feature-reweighted fine-tune, full retrain}, minimizing expected compute+carbon cost subject to an SLA-recovery constraint, using continual-learning safeguards (EWC-style regularization) to avoid forgetting when a targeted patch is chosen.

---

## SECTION 5: Core Novelty

**Publication-level novelty:**
- A new *online, model-internals-aware causal attribution algorithm* (CDAG) that fuses causal discovery with GNN embeddings of model substructure — prior causal-drift work operates purely on input feature space, never on internal activation topology.
- A formal treatment of "minimal sufficient retraining scope" as a constrained sequential decision problem, with a novel RL formulation (the RSO) whose reward jointly encodes SLA recovery probability, compute cost, and estimated carbon cost — not previously combined in this way in the continual learning literature.

**Patent-level novelty:**
- The specific method of constructing a dynamic causal graph that spans both *external data features* and *internal model activation clusters* in a single unified graph, updated online from live inference traffic.
- The counterfactual-proxy-intervention scoring technique that avoids costly ground-truth interventions (i.e., avoids actually retraining to test causality) by using a learned surrogate simulator — a concrete, technically specific method rather than an abstract idea.
- The closed-loop coupling of causal attribution output directly into a retraining-scope RL policy with carbon/cost-aware reward shaping, as an end-to-end system architecture.

---

## SECTION 6: System Architecture

```
                     ┌─────────────────────┐
   Live Traffic ───▶ │  Inference Gateway    │──▶ predictions/logs
                     └─────────┬────────────┘
                               │ feature+activation traces
                               ▼
                     ┌─────────────────────┐
                     │ Streaming Feature &   │
                     │ Activation Collector  │ (sampled, privacy-filtered)
                     └─────────┬────────────┘
                               ▼
                     ┌─────────────────────┐
                     │  CDAG Builder         │ ← incremental causal graph
                     │  (causal discovery +  │   (GNN node embeddings for
                     │   GNN embedding)       │    feature + activation nodes)
                     └─────────┬────────────┘
                               ▼
                     ┌─────────────────────┐
   Perf/SLA Monitor ▶│ Counterfactual        │
   (ground truth /   │ Attribution Engine    │ → ranked responsibility scores
    proxy labels)    └─────────┬────────────┘
                               ▼
                     ┌─────────────────────┐
                     │ Retraining Scope      │ → action: {none, partial, full}
                     │ Optimizer (RL agent)  │
                     └─────────┬────────────┘
                               ▼
                     ┌─────────────────────┐
                     │ Continual Retraining  │ (EWC-regularized subnetwork
                     │ Executor + Validator  │  fine-tune / full retrain)
                     └─────────┬────────────┘
                               ▼
                     ┌─────────────────────┐
                     │ Shadow/Canary Deploy  │ → promote or roll back
                     └─────────────────────┘
```

Supporting services: model registry, cost/carbon estimator (compute × grid-carbon-intensity API), dashboard, audit log (for compliance/explainability).

---

## SECTION 7: AI/ML Pipeline

- **Data collection**: sampled production inference logs (inputs, predictions, ground truth/proxy labels, selected internal activations at chosen "tap points").
- **Preprocessing**: PII-safe feature hashing, activation dimensionality reduction (PCA/UMAP) into cluster nodes.
- **Feature engineering**: rolling-window statistics per feature/activation cluster (mean, variance, higher moments), edge weights for the causal graph via constraint-based + score-based hybrid causal discovery (PC algorithm seeded with a NOTEARS-style continuous relaxation for scalability).
- **Model architecture**: (1) GNN encoder over the CDAG for representing node "causal centrality," (2) a lightweight learned surrogate (a small MLP/GBM) trained offline to approximate "if I intervene on node X, what's the downstream performance delta" — trained from historical retrain outcomes and synthetic interventions; (3) an actor-critic RL policy (e.g., PPO) for the Retraining Scope Optimizer.
- **Training**: surrogate and RL policy are pretrained on historical incident logs and simulated drift injections in a sandbox environment, then fine-tuned online with safe/conservative exploration (constrained by SLA guardrails).
- **Inference/deployment**: CDAG updates run as a streaming job (e.g., every N minutes); attribution + RSO decisioning run on-demand when a drift/SLA alert fires; retraining executes as an async job with shadow-mode validation before promotion.

---

## SECTION 8: Algorithms and Models

| Component | Algorithm | Why |
|---|---|---|
| Causal graph construction | Hybrid PC + NOTEARS-style continuous DAG learning | Combines interpretability of constraint-based methods with scalability of continuous optimization |
| Graph representation | Graph Neural Network (GraphSAGE/GAT) over CDAG nodes | Captures multi-hop causal influence and internal model topology jointly |
| Counterfactual scoring | Learned surrogate regressor (gradient-boosted trees or small MLP) | Avoids expensive ground-truth interventions at decision time |
| Retraining scope decision | PPO-based RL policy with constrained reward (Lagrangian penalty for SLA violation) | Naturally handles sequential, cost-sensitive decision-making under uncertainty |
| Forgetting mitigation | Elastic Weight Consolidation (EWC) / functional regularization | Prevents partial retrains from degrading unrelated model behavior |

---

## SECTION 9: Research Contribution

**Hypotheses:**
- H1: CDAG-based causal attribution identifies the true root cause of a performance drop with higher precision than correlational drift detectors (PSI/KS-test baselines).
- H2: The RSO reduces total retraining compute/carbon cost versus periodic-retrain and reactive-full-retrain baselines while maintaining equivalent or better SLA recovery time.
- H3: Coupling causal attribution to retraining scope reduces catastrophic forgetting versus naive full retrains, measured on held-out unrelated tasks/segments.

**Experiments:** synthetic drift injection benchmarks (controlled ground-truth causal factors), replayed real-world drift incidents from public MLOps postmortems, ablations removing the GNN component, the surrogate, or the RL policy (replaced with rule-based heuristics).

**Baselines:** Evidently-AI-style PSI/KS monitoring + full retrain; fixed-schedule retraining; EWC-only continual learning without causal targeting.

**Expected findings:** attribution precision/recall gains, compute-cost reduction (target: 40–70% fewer FLOPs spent on unnecessary retraining), improved SLA-recovery latency, reduced forgetting (higher retained accuracy on unrelated segments).

---

## SECTION 10: Patentability Analysis

**Potentially patentable:**
1. Method for constructing a unified causal graph spanning external data features and internal neural network activation clusters, updated incrementally from streaming inference data.
2. Method for counterfactual responsibility scoring using a learned surrogate to avoid ground-truth interventions, applied specifically to production ML drift diagnosis.
3. System and method for selecting a minimal-sufficient retraining scope via a constrained reinforcement-learning policy that jointly optimizes compute cost, carbon cost, and SLA-recovery probability.
4. The end-to-end closed-loop architecture coupling (1)+(2) into (3) with continual-learning safeguards.

**Distinction from prior art:** existing drift-monitoring patents/products cover *detection* (statistical distribution shift) but not *causal root-cause attribution reaching into model internals*, nor *cost/carbon-aware automated scope selection* as a learned policy rather than a fixed rule set.

**Possible independent claims:**
- A computer-implemented method for diagnosing performance degradation in a deployed machine learning model, comprising: constructing a causal graph including nodes representing external input features and nodes representing internal activation clusters of the model; estimating, via a trained surrogate model, a counterfactual responsibility score for each node; and selecting a retraining scope from a discrete set of options using a policy trained to minimize a cost function subject to a service-level constraint.

**Possible dependent claims:** specific graph construction method (hybrid PC/NOTEARS); GNN-based node embedding; PPO/Lagrangian-constrained RL formulation; EWC-based forgetting mitigation during scoped retraining; carbon-intensity-aware cost term.

*(Note: this is not legal advice — a patent attorney should evaluate claims and conduct formal prior-art search before filing.)*

---

## SECTION 10B: Novelty & Claims Breakdown (Detailed)

CADENCE bundles three separate novelty claims. Each should be evaluated and prior-art-searched **independently** — they don't rise or fall together, and they carry very different levels of risk.

### Claim A — Unified causal graph over external features + internal activation clusters

**What's new:** Existing causal-drift literature builds causal graphs only over input/feature space. Existing model-interpretability work (activation clustering, probing) treats internal activations as a static explanation artifact, not as live nodes in an evolving causal graph. CADENCE puts both in *one* graph and allows causal edges to run between them (e.g., feature drift → activation cluster shift → output degradation), updated incrementally from streaming production traffic.

**Why it matters:** This is the most defensible claim for patent purposes because it's a specific technical construction — not just "we combined two known techniques." It's also a genuine paper contribution: a reviewer in causal ML or ML systems would likely see this as new.

**Risk level:** Low-to-moderate. Needs a prior-art search against recent "causal graphs for model debugging" and "activation-space causal analysis" papers before relying on it in a claim.

**Draft independent claim:**
> A computer-implemented method comprising: constructing a causal graph comprising a first set of nodes representing statistical properties of external input features to a deployed machine learning model, and a second set of nodes representing statistical properties of internal activation clusters of said model; and incrementally updating edge weights of said graph from streaming inference telemetry.

---

### Claim B — Counterfactual responsibility scoring via a learned surrogate (no ground-truth intervention)

**What's new:** Applying surrogate-based counterfactual estimation specifically to score responsibility of CDAG nodes for a production performance drop, with the surrogate trained on historical retrain-outcome data as its label source.

**Why it matters:** Weakest of the three claims. Surrogate/proxy-based counterfactual effect estimation is a known idea in causal ML broadly (related to uplift modeling and causal effect estimation more generally) — the novelty here is narrow and lies entirely in the specific application, training-signal source, and object being scored (CDAG nodes), not in the general technique.

**Risk level:** High if claimed broadly. A claim written as "a method for estimating counterfactual effects via a surrogate model" will likely collide with prior art. Must be narrowly tied to the CDAG object and the retrain-outcome training signal to be defensible.

**Draft dependent claim (narrowed, attached to Claim A):**
> The method of Claim A, further comprising: training a surrogate regression model on historical retraining-outcome data to predict a performance delta resulting from a hypothetical intervention on a given node of said graph; and computing a responsibility score for each node using said surrogate model without performing said intervention.

---

### Claim C — RL-based retraining-scope policy (cost + carbon + SLA-constrained)

**What's new:** Framing "which subset of the model to retrain, if any" as a reinforcement-learning action space, with a reward that jointly penalizes compute cost, carbon cost, and SLA-violation risk. Most continual-learning research assumes the decision to retrain has already been made and only addresses *how* to retrain without forgetting. CADENCE asks *whether* and *how much* to retrain, as a learned policy rather than a fixed rule.

**Why it matters:** This is the strongest and cleanest claim in the project. It has the least overlap with both existing MLOps monitoring products (which use fixed rules or human judgment) and continual-learning literature (which starts after the retrain decision is made). This should be the anchor contribution for the paper's core claim and the primary independent patent claim.

**Risk level:** Low. Recommend building and validating this component first, since it's both the most novel and the most self-contained (it can be evaluated somewhat independently of Claims A and B using a simpler drift signal as input).

**Draft independent claim:**
> A computer-implemented method for automated model maintenance, comprising: receiving a set of responsibility scores associated with a deployed machine learning model; selecting, via a policy trained using reinforcement learning, a retraining scope from a discrete set of options including no retraining, partial subnetwork retraining, and full retraining; wherein said policy is trained to minimize a reward function comprising a compute cost term, a carbon cost term, and a constraint term penalizing violation of a service-level performance threshold.

---

### Priority order for validation and filing

1. **File/validate Claim C first** — strongest novelty, lowest prior-art risk, most self-contained to build and demonstrate.
2. **Claim A second** — solid supporting claim, moderate prior-art search needed.
3. **Claim B last, and only narrowly** — do not rely on it as a standalone claim; treat it as a dependent claim attached to Claim A, and expect to cut or narrow it further after a real prior-art search.

*Reminder: none of the above is a legal opinion. Claim language here is illustrative to guide your own drafting and prior-art search — a patent attorney should review and refine actual claims before any filing.*

---

## SECTION 11: Implementation Roadmap (6–12 months)

- **Months 1–2**: Literature review, formalize CDAG definition, build sandbox simulation environment with synthetic drift injection; implement baseline statistical drift detectors.
- **Months 3–4**: Implement CDAG builder (causal discovery + GNN embedding); validate on synthetic benchmarks against ground-truth causal factors.
- **Months 5–6**: Build counterfactual surrogate scorer; implement Retraining Scope Optimizer (start with rule-based, then PPO).
- **Months 7–8**: Integrate EWC-based continual retraining executor; build shadow/canary deployment + rollback logic.
- **Months 9–10**: End-to-end evaluation on replayed real-world drift incidents; ablation studies; begin paper draft.
- **Months 11–12**: Polish open-source release, write and submit paper (workshop-tier first), prepare patent disclosure draft, build demo dashboard for portfolio/resume.

---

## SECTION 12: Technology Stack

- **Languages**: Python (core), Go or Rust (optional for the low-latency inference gateway/collector)
- **ML/DL**: PyTorch, PyTorch Geometric (GNN), scikit-learn/LightGBM (surrogate), Stable-Baselines3 or RLlib (PPO)
- **Causal discovery**: `causal-learn`, `gcastle`, custom NOTEARS implementation
- **Streaming/orchestration**: Kafka or Redis Streams, Airflow/Prefect for retraining jobs
- **Serving**: FastAPI/Triton Inference Server, Docker, Kubernetes (or a lightweight single-node setup for the student build)
- **Storage**: PostgreSQL/TimescaleDB (metrics/time series), MLflow (model/experiment registry — she already has MLflow contribution experience)
- **Carbon/cost estimator**: CodeCarbon or Electricity Maps API
- **Monitoring/dashboard**: Grafana + Prometheus, or a custom React dashboard
- **CI/CD**: GitHub Actions

---

## SECTION 13: Evaluation Metrics

- **ML/attribution quality**: attribution precision/recall vs. known ground-truth causal factor (synthetic benchmarks), AUROC of "is this node responsible," calibration of counterfactual scores.
- **System/cost**: compute (FLOPs/GPU-hours) saved vs. full-retrain baseline, retraining decision latency, carbon footprint (kg CO2e) per incident resolved.
- **Model quality post-fix**: SLA recovery time, post-retrain accuracy/F1/AUROC, forgetting rate on unrelated held-out segments.
- **Robustness**: performance under noisy/partial telemetry, false-alarm rate.
- **Explainability**: human-rater agreement with CDAG's causal attribution on real incident postmortems.

---

## SECTION 14: Experimental Design

- **Datasets**: (1) synthetic drift-injection benchmark built on a standard tabular dataset (e.g., a shifted variant of a public credit-risk or click-through dataset) with known injected causal factors; (2) public MLOps incident postmortems/case studies replayed as evaluation scenarios; (3) an image or NLP variant to show generality (e.g., drift in a text classifier from vocabulary shift).
- **Baselines**: PSI/KS-test + full retrain; fixed weekly retrain; EWC-only continual learning without attribution.
- **Ablations**: remove GNN embedding (use raw feature stats only); remove surrogate (use naive correlation for scoring); replace RL policy with a fixed threshold rule.
- **Statistical testing**: paired t-tests / Wilcoxon signed-rank across multiple random seeds and injected-drift scenarios; report confidence intervals.
- **Reproducibility**: fixed seeds, versioned synthetic benchmark generator released publicly, Docker image for full pipeline, experiment configs tracked in MLflow.

---

## SECTION 15: GitHub Repository Structure

```
cadence/
├── README.md                  # overview, quickstart, architecture diagram
├── LICENSE
├── CONTRIBUTING.md
├── docs/                       # architecture, API reference, paper draft
├── cadence/
│   ├── collector/               # streaming feature+activation collection
│   ├── cdag/                    # causal graph builder + GNN embeddings
│   ├── attribution/             # counterfactual surrogate scorer
│   ├── rso/                     # RL-based retraining scope optimizer
│   ├── executor/                # EWC continual retraining + shadow deploy
│   └── carbon/                  # cost/carbon estimator
├── benchmarks/
│   ├── synthetic_drift_gen/     # public reproducible benchmark generator
│   └── replayed_incidents/
├── examples/                    # end-to-end notebooks/demos
├── dashboard/                   # Grafana configs or React app
├── tests/
├── docker/
├── .github/workflows/           # CI: lint, unit tests, benchmark regression
└── experiments/                 # MLflow-tracked configs and results
```

---

## SECTION 16: Research Publication Roadmap

- **Workshop-tier first** (fast feedback, low bar): NeurIPS/ICML workshop on ML systems, continual learning, or causal ML (e.g., "ML for Systems," "Distribution Shifts").
- **Full conference/journal**: extend with full ablations and real-incident case studies for an IEEE (e.g., IEEE Transactions on Services Computing / IEEE Big Data) or ACM (e.g., ACM SoCC, EuroMLSys) submission, following the compression/revision workflow already used for MOSAIC.
- **Springer target**: a systems-and-ML-focused Springer journal for the causal attribution algorithm specifically, positioned as a methods paper.
- **AAAI workshop**: a version emphasizing the causal discovery + counterfactual reasoning angle for the AAAI causal AI workshop track.

---

## SECTION 17: Startup Potential

- **Customers**: mid-to-large ML platform teams (fintech, ad-tech, health-tech, e-commerce) running many production models with real retraining cost pain.
- **Business model**: usage-based SaaS/plugin priced on GPU-hours saved or number of monitored models; enterprise tier with on-prem/VPC deployment for regulated industries.
- **Competitive advantage**: causal-level explainability ("why did the model break") is a defensible differentiator vs. purely statistical monitoring tools (Evidently, WhyLabs, Arize, Fiddler); patent-pending attribution method as a moat.
- **Market size**: sits inside the rapidly growing MLOps/observability market, where compute-cost optimization is an increasingly urgent budget line for every company running production ML.
- **Monetization**: tiered SaaS + cost-savings-based pricing ("pay a % of GPU-hours we save you") as an easy ROI pitch to enterprise buyers.

---

## SECTION 18: Resume Bullet Points

- Designed and implemented **CADENCE**, a causal attribution engine that diagnoses root causes of production ML model drift by fusing GNN-based causal graph learning with counterfactual surrogate modeling, achieving [X]% attribution precision over statistical baselines on synthetic and replayed real-world drift benchmarks.
- Built a constrained reinforcement-learning policy for automated retraining-scope decisions, reducing unnecessary retraining compute by [X]% and carbon footprint by [X]% while maintaining SLA recovery time within [X]% of full-retrain baselines.
- Architected and open-sourced an end-to-end MLOps system integrating streaming telemetry collection, causal graph construction, EWC-regularized continual retraining, and shadow-deployment validation, deployed and benchmarked across [N] simulated production incidents.

---

## SECTION 19: Interview Questions

1. Why is correlational drift detection (PSI/KS-test) insufficient for root-cause diagnosis, and how does causal attribution address that gap?
2. Walk through how the CDAG is constructed — why hybrid PC + NOTEARS rather than either alone?
3. How do you avoid the chicken-and-egg problem of needing interventions to establish causality without actually performing costly retrains?
4. What guarantees (if any) does the surrogate counterfactual scorer have, and how do you validate its calibration?
5. Why frame retraining-scope selection as an RL problem rather than a supervised classification problem?
6. How does the Lagrangian constraint in the RL reward prevent SLA violations during exploration?
7. Explain how EWC prevents catastrophic forgetting during a partial/subnetwork retrain — what are its limitations?
8. How do you scale causal discovery to high-dimensional production feature spaces in near real-time?
9. What privacy considerations arise from tapping internal model activations, and how do you mitigate them?
10. How would you detect and handle a case where the causal graph itself is wrong (confounded features)?
11. How do you evaluate attribution quality without ground-truth causal labels in real production data?
12. What's the failure mode if the surrogate model is systematically biased toward under- or over-estimating responsibility?
13. How does this system differ architecturally from Evidently AI / WhyLabs / Arize, and where would it fail to differentiate?
14. How would you extend CADENCE to multimodal or LLM-based production systems?
15. What are the compute/latency trade-offs of running this attribution pipeline continuously vs. only on-alert?

---

## SECTION 20: Future Research Directions (PhD-level extensions)

1. **Federated causal attribution**: extending CDAG construction across federated/edge fleets without centralizing raw activation data, using privacy-preserving causal discovery.
2. **Multimodal causal graphs**: unifying causal attribution across text, image, and tabular modalities for multimodal foundation model drift diagnosis.
3. **Theoretical guarantees for surrogate-based counterfactual scoring**: bounding the approximation error between the learned surrogate and true interventional effects.
4. **LLM-specific drift attribution**: adapting CDAG to attribute behavioral drift in large language models (e.g., prompt-distribution shift, tool-use failure attribution) rather than classical tabular/vision models.
5. **Self-improving causal graphs**: meta-learning the causal discovery algorithm itself from accumulated incident histories across many deployed models, forming a continually improving "causal prior library" for MLOps.

---

*Fits naturally alongside your MOSAIC (systems/fairness) and vramfit (scheduling) work — this project extends into causal ML + continual learning while staying in the systems-for-ML lane you've been building a track record in.*
