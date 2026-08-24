# CADENCE — Master Reference Document

This is the single, merged reference for the CADENCE project, combining three previously separate files:

1. **Part I — Project Proposal**: the full 20-section research/patent-depth proposal.
2. **Part II — Explained From Zero**: the beginner-level, ground-up walkthrough using the FraudNet example (46-question breakdown, numerical example, per-model tables, data-flow diagram, partial/full retraining mechanics, and the honest gap analysis).
3. **Part III — Dataset Reference**: the complete, verified dataset/link list, organized by what each dataset validates.

Use the table of contents below to jump to any section. Nothing from any of the three source files has been shortened or omitted — this is a direct merge.

---

## Table of Contents

**Part I — Project Proposal**
- Sections 1-20 (title through future research directions), including Section 10B (Novelty & Claims Breakdown)

**Part II — Explained From Zero**
- Part 1: The 46 Questions, Answered Concretely
- Part 2: One Complete Numerical Walkthrough
- Part 3: Per-Model Tables (A-F)
- Part 3B: Partial vs. Full Retraining - The Exact Mechanics
- Part 4: How Many ML Models Are Actually Being Trained?
- Part 5: Complete Data-Flow Diagram
- Part 6: Honest Gap Analysis

**Part III — Dataset Reference**
- Production model datasets by model type
- Synthetic drift generators
- Real-world drift datasets
- Continual-learning benchmarks
- Text/NLP generality datasets
- Real incident case studies
- Carbon/cost estimation tools
- Quick-reference table
- Where to start given your hardware

---

# PART I — PROJECT PROPOSAL

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

---

# PART II — CADENCE, EXPLAINED FROM ZERO

# CADENCE, Explained From Zero — One Example, Start to Finish

**Ground rule for this document:** everything here uses the CADENCE proposal as source of truth for *what* algorithms are used (hybrid PC+NOTEARS causal discovery, GraphSAGE/GAT-style GNN, MLP/GBM surrogate, PPO with a constrained reward, EWC). Where the original proposal was high-level ("a GNN encoder," "a learned surrogate"), this document fills in a **concrete, illustrative** version so you can see exactly what would be built — those concrete choices are clearly marked as decisions we're now making, not things the original proposal already specified. Section 12 at the end lists precisely which is which.

**Running example used throughout:** a credit-card fraud detection neural network.

---

## PART 1 — The 46 Questions, Answered Concretely

### 1. What does the original dataset look like?
A table of past credit card transactions. Each row = one transaction. Columns: `Time` (seconds since first transaction), `Amount` (transaction value), `V1`...`V28` (28 anonymized numeric features, e.g. PCA-transformed raw transaction details for privacy), and a label `Class` (1 = fraud, 0 = legitimate). Realistically ~280,000 rows, heavily imbalanced (fraud is rare — maybe 0.2% of rows).

### 2. What is the production model?
**FraudNet** — a small feedforward neural network that takes one transaction's features and outputs a fraud probability.

### 3. What exact architecture does FraudNet use?
```
Input (30 features: Time, Amount, V1–V28)
   → Dense layer, 64 neurons, ReLU activation      [Layer 1]
   → Dense layer, 32 neurons, ReLU activation      [Layer 2]
   → Dense layer, 1 neuron, Sigmoid activation     [Output]
```
This is deliberately a small, simple MLP — CADENCE is designed for exactly this scale of model (see the scope discussion from earlier: small/mid classifiers, not LLMs).

### 4. What features go in?
The same 30 columns from the dataset (minus the label): `Time`, `Amount`, `V1`...`V28`.

### 5. What does the model output?
A single number between 0 and 1 — the estimated probability the transaction is fraudulent. In production this is thresholded (e.g. `> 0.5` → flag as fraud) to produce a yes/no decision.

### 6. How is FraudNet initially trained?
Standard supervised learning: feed it labeled historical transactions, compare its predicted probability to the true label (0 or 1) using a loss function, and adjust its weights via backpropagation until it performs well on a held-out test set. (Full details in Table A, Part 3.)

### 7. What does "production" mean here?
FraudNet is deployed behind a live scoring service. Every real transaction that comes in gets fed through FraudNet in real time, and it returns a fraud probability used to approve/flag the transaction. This is the ongoing, continuous phase CADENCE watches.

### 8. What data does CADENCE collect after deployment?
For every scored transaction (or a sampled subset, for efficiency): the input feature vector, FraudNet's predicted probability, the eventual true label if/when it becomes known (e.g. a chargeback confirms fraud days later), and the internal activation values at Layer 1 and Layer 2 (see next question) for that transaction.

### 9. What is an "internal activation"?
It's the vector of numbers a hidden layer produces for a given input — not the final prediction, but the intermediate "thinking." For one transaction, Layer 1's activation is a 64-number vector (one number per neuron in that layer); Layer 2's is a 32-number vector.

### 10. Why do we collect activations at all?
Because CADENCE's core idea is that drift can show up **inside** the model, not just in the raw input data. If a specific pattern of transactions starts triggering a specific subset of Layer 1's neurons very differently than during training, that's a signal — one that raw-feature monitoring alone wouldn't necessarily catch. It's also what lets CADENCE later say "retrain *this* layer/subnetwork" instead of only "retrain everything."

### 11. What exactly does "model drift" mean here?
It means: the statistical relationship between the inputs and the correct output has changed since FraudNet was trained, causing its real-world accuracy to degrade — even though FraudNet itself hasn't changed at all. Concretely: FraudNet's F1 score on live traffic (measured once true labels arrive) drops compared to its original test-set F1.

### 12. How do we deliberately create drift for experimentation?
We can't wait for real fraud patterns to shift, so during development we inject a **known, controlled** change into a copy of the production data stream — e.g., artificially shift the `Amount` column's distribution upward (simulating fraudsters moving to higher-value transactions), or flip the fraud-generating rule for a specific value range of `V14`. This is exactly what the `river` synthetic drift generators are used for (see Section 5 of the earlier proposal for these tools).

### 13. How do we know the *true* cause of the drift?
Only because **we** injected it ourselves and logged exactly what we changed and by how much. This becomes the answer key: later, when CDAG guesses "I think `Amount` is responsible," we can check that guess against the fact we already know is true. In real production, this ground truth is never available — that's precisely why the attribution problem is hard and worth solving.

### 14. How is drift *detected* in the first place (the trigger)?
A statistical monitor compares recent production feature distributions (and/or recent performance, once labels arrive) against the training-time baseline, using a standard test — e.g. Population Stability Index (PSI) or a Kolmogorov–Smirnov test — on a rolling window. When the divergence crosses a threshold, or measured F1 drops below a target, an alert fires. **This is a separate, simpler mechanism from the CDAG** — it's the smoke alarm; CDAG is the detective who shows up after the alarm goes off.

### 15. What happens after drift is detected?
The alert triggers the rest of the pipeline: CDAG is built/updated from recent data, the attribution engine scores which node(s) are responsible, the PPO agent decides a retraining scope, the fix is applied and validated, and (if it passes) redeployed.

### 16. What is a "causal graph" in this project?
The **CDAG (Causal Drift Attribution Graph)**: a graph where nodes represent things that could be responsible for a performance change, and directed edges represent a believed cause→effect relationship between them, learned from data rather than hand-specified.

### 17. What does every node in the CDAG represent?
Three kinds of nodes:
- **External feature nodes** — one per input feature's rolling statistics, e.g. a node for "`Amount`'s windowed mean/variance," a node for "`V14`'s windowed mean/variance," etc. (30 such nodes for our example, one per feature.)
- **Internal activation-cluster nodes** — because a raw 64-dim or 32-dim activation vector is too large to use as individual nodes, we first group each layer's neurons into a small number of clusters (e.g., via k-means on neuron activation patterns from training data) and treat each cluster's average activity as one node. E.g., 8 cluster nodes for Layer 1, 4 for Layer 2.
- **One outcome node** — the model's rolling performance metric (e.g., windowed F1 or error rate).

### 18. What does every edge represent?
A directed, weighted causal link — e.g. an edge from `Amount_node → L1_cluster3_node` with weight 0.6 means "shifts in `Amount`'s distribution are estimated to causally drive about 60% of the variation seen in Layer 1 cluster 3's activity," and further edges eventually connect into the outcome node (`... → Performance_node`).

### 19. How does PC contribute to the graph?
The PC algorithm runs conditional-independence tests (e.g., partial correlation tests) between every pair of nodes' recent time-windowed statistics. If two nodes remain statistically dependent even after conditioning on all other nodes, PC keeps a candidate edge between them; otherwise it removes it. This produces a **sparse skeleton** — which edges are even allowed to exist — plus some edge orientation.

### 20. How does NOTEARS contribute to the graph?
NOTEARS takes the same node time-series data and fits a **continuous, weighted adjacency matrix** by minimizing a reconstruction error (how well each node's value is predicted from its parent nodes) while smoothly enforcing "no cycles allowed" via a differentiable acyclicity constraint. This gives real-valued edge *weights*, and can find structure PC might miss with limited samples.

### 21. Why do we need both?
PC is interpretable and statistically principled but scales poorly and needs a fair amount of data per test to be reliable; used alone it can miss real edges when production samples are limited. NOTEARS scales well and gives usable continuous weights, but used alone (without a well-chosen sparsity penalty) it can produce a dense graph full of spurious edges. The hybrid approach: **use PC first to decide which edges are allowed to exist (the skeleton)**, then **run NOTEARS's continuous optimization restricted to only that skeleton** to assign real-valued weights. You get PC's discipline and NOTEARS's usable weights.

### 22. What does the GNN receive as input?
The CDAG graph structure itself: for every node, a feature vector (recent windowed statistics — mean, variance, skew — for feature nodes; cluster-centroid distance-from-baseline for activation nodes); and for every edge, its weight from step 21.

### 23. What does the GNN output?
A learned embedding vector for every node (e.g., a 16-number vector per node) that summarizes not just that node's own statistics, but the influence flowing in from its causal neighbors (multi-hop context).

### 24. What is the GNN actually trained to learn?
To produce node embeddings that are *useful for predicting how much fixing that node would help* — i.e., embeddings from which the downstream surrogate (next section) can accurately predict counterfactual recovery. The GNN isn't given a separate, standalone objective; it's trained jointly with the surrogate, end-to-end, so its embeddings are shaped entirely by that downstream task.

### 25. What are the GNN's training labels?
The same labels the surrogate uses (see #29) — actual measured post-intervention performance from historical/simulated incidents. The GNN has no independent label of its own; gradients flow back into it from the surrogate's loss.

### 26. What loss function does the GNN use?
Mean Squared Error (MSE), shared with the surrogate — the GNN+surrogate is really one trainable pipeline with two conceptual stages, trained with a single regression loss end-to-end.

### 27. What does the counterfactual surrogate receive as input?
The GNN embedding for a specific candidate node, plus some context features: the current magnitude of the performance drop, and the current windowed statistics.

### 28. What is the surrogate's output?
A single predicted number: "if this node's factor were corrected/reverted to its training-time baseline, the model's performance would likely be **this** value" (e.g., predicted F1 = 0.91).

### 29. What are the surrogate's training labels?
The **actual, measured** post-intervention performance from real or simulated "we fixed this and here's what happened" events.

### 30. Where do those labels come from?
Two sources, generated **offline**, before the surrogate ever sees live production traffic:
- **Synthetic sandbox experiments**: inject a known drift (per #12), actually retrain/patch the sandboxed model to fix that specific cause, and measure the real resulting performance. Repeat across many synthetic scenarios.
- **Historical real incidents**: past cases where an engineer diagnosed a real cause and fixed it, with before/after performance logged.

### 31. How do we generate counterfactual training examples without retraining the *live production* model every time?
This is the key trick: all of the expensive "actually retrain and measure" work happens **once, offline, during the surrogate's own training phase**, inside a sandbox — never against the live model. Once the surrogate has learned this mapping from many such offline examples, at live inference time it just does a fast forward pass (predict, don't retrain) to estimate what fixing a given node would do — no real retraining needed to get that estimate.

### 32. What does "responsibility score" mathematically mean?
It's the *predicted improvement* from fixing a given node:
`responsibility_score(node) = surrogate_predicted_F1_if_node_fixed − current_observed_F1`
A higher score means fixing that node is predicted to recover more performance — i.e., that node is estimated to be more responsible for the current drop.

### 33. How is the responsibility score calculated, step by step?
1. Build/update the CDAG (steps 16–21).
2. Run the GNN to get an embedding per node (steps 22–26).
3. For each candidate node, feed its embedding + context into the surrogate to get a predicted post-fix F1 (steps 27–30).
4. Subtract the current observed F1 from each predicted value.
5. Rank all nodes by this score — highest score = most responsible.

### 34. What does the PPO agent receive as its state?
A vector summarizing the current situation: the top-k responsibility scores and which nodes they belong to, current performance vs. SLA target, how long the drift has persisted, estimated compute cost for each possible action (no-op/partial/full), current carbon-intensity of the grid (from an API), and time since the model was last retrained.

### 35. What actions can PPO take?
A discrete choice among exactly three options: **no-op** (do nothing), **partial retrain** (fine-tune only the subnetwork/layer tied to the most responsible node), or **full retrain** (retrain the entire model).

### 36. What is the reward function?
Conceptually:
`Reward = (performance recovery achieved) − w1·(compute cost) − w2·(carbon cost) − penalty·(if SLA is violated after the action)`
where the penalty term is enforced via a Lagrangian-style constraint (the earlier proposal's Claim C design) so the policy is discouraged from choosing cheap options that leave performance unacceptably low.

### 37. Where does the reward actually come from?
During PPO's own training, the reward is computed inside the **simulated sandbox environment** — after a simulated action, the simulator tells PPO the resulting performance, the compute time it would have taken, and the carbon estimate for that compute (from a tool like CodeCarbon). During real operation later, the reward can be computed from the real outcome after shadow/canary validation (see #44–45), potentially used to keep fine-tuning the policy conservatively.

### 38. What is PPO actually learning?
A policy — a function from "situation" (state) to "best action" — that, through repeated trial and error against the reward signal, learns *when* a no-op is safe, *when* a partial fix is sufficient, and *when* a full retrain is truly justified, balancing cost/carbon against the risk of leaving performance too low.

### 39. Why PPO instead of a simple rule-based system (e.g., "if responsibility > 0.8, do a partial retrain")?
Because the *right* decision depends on several interacting factors that a fixed rule can't easily balance — how confident/concentrated the attribution is, how expensive compute is right now, how close performance is to violating the SLA, how long it's been since the last retrain. A hand-written rule would need to be re-tuned constantly and can't learn from outcomes. PPO learns this balance from experience and can adapt as conditions change (e.g., learning that acting fast matters more when the SLA margin is thin).

### 40. What does EWC do?
EWC (Elastic Weight Consolidation) adds a penalty term to the retraining loss that discourages parameters from moving far away from their pre-retrain values — weighted by how *important* each parameter was estimated to be for previously-learned behavior (estimated via the Fisher information matrix). This is what stops a "small fix" from accidentally breaking things the model used to do well.

### 41. When is EWC used?
Only when PPO chooses **partial retrain**. It's not used for a full retrain (which is a fresh, complete training run) and obviously not for a no-op.

### 42. What parameters actually get retrained during partial retraining?
Only the weights belonging to the subnetwork/layer tied to the most-responsible node from CDAG. In our example, if the responsible node is an activation cluster in **Layer 1**, only Layer 1's weights are updated; Layer 2 and the output layer are frozen (or only lightly regularized), with the EWC penalty applied to discourage even Layer 1's weights from drifting too far from where they were.

### 43. What happens during full retraining?
FraudNet's entire architecture — all layers — is retrained (either from scratch or warm-started from current weights) on the combined historical + new production-labeled data, the same way it was originally trained (Table A), without any subnetwork localization or EWC constraint.

### 44. What happens during shadow/canary validation?
The freshly retrained (partial or full) model is deployed **alongside** the current production model, scoring the same live traffic (or a recent held-out validation window with delayed labels) without its predictions being used for real decisions yet. Its performance is measured over that window. If it meets or exceeds a recovery threshold (e.g., F1 ≥ SLA target), it's gradually rolled out — first to a small percentage of real traffic (canary), then fully, if it continues to perform well.

### 45. What happens if validation fails?
The new model is **rolled back** — the old production model keeps serving all traffic, the failure is logged, and (importantly) this outcome is fed back as a new training example to the surrogate ("we predicted 0.91, but the real fix only achieved 0.83") so the surrogate's calibration improves over time. A human engineer may also be escalated to for cases the automated pipeline can't resolve.

### 46. How does the system loop back and keep monitoring?
After a successful promotion (or a rollback), CADENCE simply resumes its normal continuous state: the drift monitor (#14) keeps watching new production traffic, the CDAG keeps incorporating new windowed statistics, and the whole cycle is ready to trigger again the next time a drift alert fires.

---

## PART 2 — One Complete Numerical Walkthrough

| Step | What happens | Numbers |
|---|---|---|
| 1. Baseline | FraudNet trained and validated | F1 = **0.94** |
| 2. Injected drift (for experimentation) | `Amount`'s mean shifts from training baseline | Baseline mean $88 → production mean $135 (+53%) |
| 3. Consequence | FraudNet's live F1 degrades | F1 drops to **0.79** |
| 4. Internal effect | Layer 1's activation cluster #3 (the cluster most sensitive to `Amount`) shifts | Cluster centroid distance-from-baseline: 0.02 → **0.31** |
| 5. Drift detection trigger fires | PSI on `Amount` crosses threshold; F1 also below SLA (0.90) | Alert fires |
| 6. CDAG built/updated | Nodes: `Amount`, `Time`, `V1`...`V28`, 8 L1-cluster nodes, 4 L2-cluster nodes, 1 performance node | — |
| 7. GNN + surrogate attribution | Responsibility scores computed per node | `Amount` node = **0.83**, next-highest (`Time`) = **0.11**, all others < 0.05 |
| 8. Surrogate's predicted recovery | If `Amount`-linked subnetwork (Layer 1) is fixed | Predicted F1 = **0.91** |
| 9. PPO decision | High, concentrated responsibility (0.83) + predicted recovery (0.91) clears SLA (0.90) at much lower cost than full retrain | Action chosen: **partial retrain** (Layer 1 only) |
| 10. Cost comparison (from state features) | Partial vs. full | Partial: **0.5 GPU-hr**, ~0.05 kg CO₂ · Full: **6.0 GPU-hr**, ~0.6 kg CO₂ |
| 11. EWC-regularized fine-tune of Layer 1 | Layer 2 + output frozen, Fisher-weighted penalty applied | — |
| 12. Shadow/canary validation | New model tested on held-out recent window | Measured F1 = **0.92** (slightly beats the 0.91 prediction) |
| 13. Promotion decision | 0.92 ≥ SLA (0.90) → passes | Promoted: canary rollout → full rollout |
| 14. Compute saved vs. always doing a full retrain | (6.0 − 0.5) / 6.0 | **≈ 91.7% compute reduction** |
| 15. System resumes monitoring | Back to step 5's watching state | — |

---

## PART 3 — Per-Model Tables

### A. Production Model (FraudNet)

| Field | Value |
|---|---|
| Purpose | Predict fraud probability for a transaction |
| Input | 30-dim feature vector (`Time`, `Amount`, `V1`–`V28`) |
| Output | Scalar fraud probability, 0–1 (thresholded at 0.5 for a decision) |
| Training dataset | Historical labeled transactions (~280K rows, ~0.2% fraud) |
| Training labels | Ground-truth `Class` (0/1), from confirmed fraud/chargeback records |
| Loss function | Binary cross-entropy (class-weighted, given imbalance) |
| Optimizer | Adam, learning rate ≈ 1e-3 |
| Trainable parameters | ~4,100 (weights + biases across the 3 dense layers) |
| Hyperparameters | Batch size 256; up to ~30 epochs with early stopping; dropout 0.2 |
| When training happens | Once initially; again (fully or partially) whenever CADENCE triggers a retrain |
| When inference happens | Continuously, in real time, on every live transaction |
| Evaluation metrics | F1, AUROC, precision, recall (imbalance makes plain accuracy misleading) |

### B. Causal Discovery (Hybrid PC + NOTEARS) — *a re-fit procedure, not a persistent trained model*

| Field | Value |
|---|---|
| Purpose | Determine which nodes are causally connected (skeleton) and how strongly (weights) |
| Input | Rolling windowed statistics for every node (features, activation clusters, performance) |
| Output | A weighted, directed adjacency structure (the CDAG) |
| "Training dataset" | Recent windowed production statistics, recomputed each monitoring cycle |
| "Training labels" | None (unsupervised) — PC uses conditional-independence tests; NOTEARS uses a self-supervised reconstruction objective |
| Loss/objective | PC: statistical independence test thresholding. NOTEARS: squared reconstruction error + acyclicity constraint + L1 sparsity penalty |
| Optimizer | NOTEARS stage: augmented Lagrangian with L-BFGS or Adam on the continuous relaxation |
| "Trainable parameters" | The weighted adjacency matrix W (size = nodes × nodes), re-fit each cycle, not persisted like a model checkpoint |
| Hyperparameters | PC significance level α; NOTEARS sparsity penalty λ; acyclicity tolerance |
| When it runs | Periodically (each monitoring window) and on-demand when a drift alert fires |
| Evaluation metrics | Structural Hamming Distance vs. known ground truth (measurable only on synthetic benchmarks where the true graph is known) |

### C. GNN (CDAG-GNN)

| Field | Value |
|---|---|
| Purpose | Produce contextualized per-node embeddings capturing multi-hop causal influence |
| Input | CDAG structure: per-node feature vectors + per-edge weights from B |
| Output | Per-node embedding vector (e.g., 16-dim) |
| Training dataset | Same sandbox/historical intervention dataset as the surrogate (D) — trained jointly, end-to-end |
| Training labels | Same as D's labels (actual post-intervention performance) — no independent label of its own |
| Loss function | MSE (shared with D, backpropagated through both) |
| Optimizer | Adam |
| Trainable parameters | GraphSAGE/GAT-style message-passing and aggregation weights |
| Hyperparameters | Number of layers (e.g., 2), hidden dim (e.g., 32), output embedding dim (16), learning rate |
| When training happens | Offline, periodically refreshed as more incident data accumulates |
| When inference happens | Every time an attribution cycle is triggered by a drift alert |
| Evaluation metrics | No standalone metric — evaluated jointly via the surrogate's downstream accuracy |

### D. Counterfactual Surrogate

| Field | Value |
|---|---|
| Purpose | Predict the performance that would result if a given node's cause were fixed |
| Input | GNN embedding for a candidate node + context (current drop magnitude, window stats) |
| Output | Scalar predicted post-fix performance (e.g., predicted F1) |
| Training dataset | Offline sandbox: synthetic drift-injection + real-fix experiments; plus historical real-incident before/after records |
| Training labels | Actual measured post-intervention performance from those offline experiments |
| Loss function | MSE (predicted vs. actual post-intervention performance) |
| Optimizer | Adam (if MLP) or built-in boosting optimizer (if gradient-boosted trees) |
| Trainable parameters | MLP weights, or a GBM tree ensemble, depending on final architecture choice |
| Hyperparameters | Tree depth/count (GBM) or hidden units/layers (MLP); learning rate |
| When training happens | Offline, periodically, as more sandbox/incident data accumulates |
| When inference happens | Once per candidate node, every time a drift alert triggers attribution |
| Evaluation metrics | MSE/MAE of prediction vs. actual outcome; calibration (does a 0.91 prediction average out to ~0.91 in reality) |

### E. PPO Agent (Retraining Scope Optimizer)

| Field | Value |
|---|---|
| Purpose | Decide retraining scope (no-op / partial / full) under cost, carbon, and SLA constraints |
| Input (state) | Top-k responsibility scores, current performance, drift duration, per-action compute-cost estimate, current grid carbon intensity, SLA threshold, time since last retrain |
| Output (action) | One of: no-op, partial retrain, full retrain |
| Training dataset | Simulated episodes in a sandbox (Gym-style) environment built from synthetic drift injections + cost/carbon models |
| Training labels | None (reinforcement learning, not supervised) |
| Loss function | PPO clipped surrogate objective + value function loss + entropy bonus, with a Lagrangian penalty term for the SLA constraint |
| Optimizer | Adam |
| Trainable parameters | Policy network weights + value network weights |
| Hyperparameters | Clip ratio ε, discount γ, GAE λ, learning rate, Lagrangian multiplier update rate |
| When training happens | Offline in simulation initially; can be conservatively fine-tuned online as real incidents accumulate |
| When inference happens | Every time a drift alert fires and attribution scores are available |
| Evaluation metrics | Cumulative reward, compute/carbon saved vs. baselines, SLA violation rate, average recovery time |

### F. EWC / Continual Retraining Executor — *a technique applied to Model A, not a separate model*

| Field | Value |
|---|---|
| Purpose | Prevent catastrophic forgetting while partially retraining FraudNet |
| Input | Current FraudNet weights, Fisher information (parameter importance from prior training), recent production data + a small replay buffer of historical data |
| Output | Updated weights — only for the targeted subnetwork/layer; others frozen |
| "Training dataset" | Recent post-drift production window + replay-buffer sample of historical training data |
| "Training labels" | True (or best-available proxy) fraud labels for that recent window |
| Loss function | Task loss (BCE, same as A) + EWC penalty: Σ Fisher_i · (θ_i − θ*_i)² over the affected parameters |
| Optimizer | Adam, smaller learning rate than initial training (e.g., 1e-4) |
| Trainable parameters | Only the identified subnetwork's weights (e.g., Layer 1 only in our example) |
| Hyperparameters | EWC penalty weight λ, fine-tune LR, number of fine-tune epochs, replay buffer size/ratio |
| When training happens | Triggered specifically by PPO choosing "partial retrain" |
| When inference happens | N/A directly — produces an updated FraudNet, whose inference is covered by Table A |
| Evaluation metrics | Post-retrain F1; forgetting rate (accuracy on unrelated historical segments, before vs. after) |

---

## PART 3B — Partial vs. Full Retraining: The Exact Mechanics

Starting point in all three cases: CDAG + GNN + surrogate have already produced ranked responsibility scores, and PPO has already picked an action. From here, the Retraining Executor takes over.

### Case 1: No-op
Nothing is retrained. The executor logs the decision (which node was flagged, its score, and why PPO judged it not worth acting on), and the system returns straight to monitoring.

### Case 2: Partial retrain — step by step

**Step 1 — Translate "responsible node" into "which weights to touch."**
The most-responsible node points to a specific layer (e.g., Layer 1), because each activation-cluster node was created *from* that layer's neurons in the first place — so the mapping back from node → layer is a direct lookup, not a learned step.

**Step 2 — Freeze everything except the target layer.**
```python
for name, param in fraudnet.named_parameters():
    if "layer1" not in name:      # target layer identified in Step 1
        param.requires_grad = False
    else:
        param.requires_grad = True
```

**Step 3 — Compute EWC importance weights (Fisher information) for the target layer, using the *original* training data (or a cached sample of it), before the fine-tune starts.**
```python
fisher = {}
for x, y in original_training_sample:      # small representative sample
    fraudnet.zero_grad()
    output = fraudnet(x)
    loss = bce_loss(output, y)
    loss.backward()
    for name, param in fraudnet.named_parameters():
        if "layer1" in name:
            fisher[name] = fisher.get(name, 0) + param.grad.data ** 2
for name in fisher:
    fisher[name] /= len(original_training_sample)
```
High Fisher value = important parameter, protect it. Low value = safe to move.

**Step 4 — Assemble the fine-tuning dataset: recent drifted data + a replay buffer of old data.**
```python
finetune_data = recent_production_data_with_labels + replay_buffer_sample(historical_data, size=N)
```
The replay buffer matters: without it, the target layer could overfit to the new distribution alone — a second, subtler way to cause forgetting even with EWC active. A typical starting ratio is roughly 70% new data / 30% replayed old data, tuned empirically.

**Step 5 — Fine-tune the target layer with a combined loss.**
```python
for epoch in range(num_finetune_epochs):
    for x, y in finetune_data:
        pred = fraudnet(x)
        task_loss = bce_loss(pred, y)

        ewc_penalty = 0
        for name, param in fraudnet.named_parameters():
            if name in fisher:
                ewc_penalty += (fisher[name] * (param - theta_star[name]) ** 2).sum()

        total_loss = task_loss + lambda_ewc * ewc_penalty
        total_loss.backward()
        optimizer.step()   # only updates the target layer (requires_grad=False elsewhere)
```
`theta_star[name]` is the target layer's weight snapshot taken right before fine-tuning began (Step 2). The EWC penalty is a tug-of-war between fitting the new data (`task_loss`) and not drifting far from prior behavior, especially on important weights (`ewc_penalty`).

**Step 6 — Output.** An updated FraudNet where only the target layer's weights changed; all other layers are bit-for-bit identical to before. This updated model proceeds to shadow/canary validation — it is not yet serving real traffic.

### Case 3: Full retrain — mechanically simpler

```python
for param in fraudnet.parameters():
    param.requires_grad = True   # nothing frozen

training_data = full_historical_dataset + all_new_labeled_production_data
optimizer = Adam(fraudnet.parameters(), lr=1e-3)   # optionally warm-start from current weights

for epoch in range(num_epochs):
    for x, y in training_data:
        pred = fraudnet(x)
        loss = bce_loss(pred, y)
        loss.backward()
        optimizer.step()
```
No EWC, no freezing, no replay-buffer balancing — this is the original training procedure (Table A) run again on the enlarged dataset. Simpler to implement, but that simplicity is exactly why it costs more compute (Part 2's example: 6.0 GPU-hours vs. 0.5 for the partial route) — you pay for the full search space every time instead of constraining it to where the problem actually is.

### A real gap worth flagging: what if more than one layer looks responsible?

The mechanics above assume a single, clearly-dominant responsible node/layer — true in the clean numerical example in Part 2, but not guaranteed in real drift events. A production version needs an explicit rule for this — e.g., "unfreeze every layer whose node responsibility score exceeds a threshold τ" rather than defaulting to always just the single top node. Diffuse responsibility spread across many nodes might itself be a signal that the problem isn't localized and a full retrain is actually the right call. This threshold, and the diffuse-responsibility fallback behavior, is not yet specified — it belongs in Part 6's "need to decide" list alongside the other open items.

---

A direct count, no hedging:

**4 distinct trained ML models/components:**
1. **FraudNet** (the production model) — a genuine, persistent trained neural network, retrained (fully or partially) over time.
2. **CDAG-GNN** — a genuine trained neural network (graph neural net).
3. **Counterfactual Surrogate** — a genuine trained ML model (MLP or GBM); trained jointly/end-to-end with #2, but architecturally a distinct component with its own output head.
4. **PPO Agent** (policy network + value network) — a genuine trained reinforcement-learning model.

**Plus 1 recurring statistical procedure that is *not* a persistent trained model in the usual sense:**
5. **PC + NOTEARS causal discovery** — this *fits* a weighted adjacency matrix each monitoring cycle via optimization, but it's re-fit from scratch (or updated incrementally) each time rather than being a checkpointed model you load and reuse the way you would #1–#4. Whether you count this as a "5th model" is genuinely a matter of definition — worth being upfront about in an interview rather than picking a number and hiding the ambiguity.

**Not a separate model at all:**
- **EWC** is a regularization *technique* applied during Model #1's retraining step — it has no independent parameters, dataset, or output of its own.

**Bottom line to say out loud:** *"CADENCE trains four distinct ML models — the production classifier, a GNN, a counterfactual surrogate, and a PPO policy — plus a graph-fitting procedure that's re-computed rather than persistently trained, and one regularization technique (EWC) applied to the production model's own retraining."*

---

## PART 5 — Complete Data-Flow Diagram, Arrow by Arrow

```
[Historical labeled transactions]
        │  (features + true fraud labels)
        ▼
[Production Model Training]  ──── trains ───▶  FraudNet (Table A)
        │
        │  deployed weights
        ▼
[Deployment]  ──▶  FraudNet now serves live traffic
        │
        │  every incoming transaction (features only)
        ▼
[Production Traffic / Live Scoring]
        │
        │  logged: (features, predicted probability, internal activations,
        │           true label once/if it arrives later)
        ▼
[Drift Detection Trigger]  (PSI/KS-test on rolling windows, or F1 drop)
        │
        │  fires an alert containing: which window, current performance
        ▼
[CDAG Builder: PC + NOTEARS]
        │  input: rolling windowed stats for every feature node,
        │         every activation-cluster node, and the performance node
        │  output: a weighted, directed graph (the CDAG)
        ▼
[GNN]
        │  input: CDAG structure (node feature vectors + edge weights)
        │  output: a learned embedding vector per node
        ▼
[Counterfactual Surrogate]
        │  input: each node's embedding + current drop context
        │  output: predicted post-fix performance per node
        │          → subtracted from current F1 → responsibility scores
        ▼
[PPO Agent]
        │  input (state): responsibility scores, current F1, SLA target,
        │                  per-action compute/carbon cost estimates,
        │                  time since last retrain
        │  output (action): no-op / partial retrain / full retrain
        ▼
[Retraining Executor]
        │  if partial: apply EWC-regularized fine-tune to the identified
        │              subnetwork only, using recent data + replay buffer
        │  if full: retrain the entire FraudNet from historical + new data
        │  if no-op: skip straight to "resume monitoring" below
        ▼
[EWC-Regularized Fine-Tune]  (only when "partial" was chosen)
        │  output: updated weights for the targeted subnetwork
        ▼
[Shadow / Canary Validation]
        │  input: new model's predictions on held-out recent traffic
        │  output: pass/fail decision vs. SLA recovery threshold
        ▼
   ┌────┴─────┐
 pass         fail
   │            │
   ▼            ▼
[Promote to   [Rollback: keep old FraudNet serving;
 Production]   log outcome as a new (predicted vs. actual)
   │            training example for the Surrogate]
   │            │
   └─────┬──────┘
         ▼
[Resume Monitoring]  ──▶  loops back to [Production Traffic / Live Scoring]
```

---

## PART 6 — Honest Gap Analysis

### What the original CADENCE proposal already defined
- The overall pipeline stages and their order (Sections 6–7 of the proposal): collector → CDAG builder → attribution engine → retraining-scope optimizer → executor → shadow/canary deploy.
- The **algorithm family** chosen for each stage (Section 8's table): hybrid PC + NOTEARS for causal discovery; a GraphSAGE/GAT-style GNN; an MLP or GBM surrogate; a PPO policy with a Lagrangian-constrained reward; EWC for forgetting mitigation.
- The **three novelty claims** and draft patent claim language (Section 10B).
- General categories of evaluation metrics (Section 13) and experimental design at a high level (Section 14).
- The general shape of the reward function (compute cost + carbon cost + SLA constraint), conceptually.

### What the proposal did *not* specify — real gaps
- **Production model architecture** — deliberately left generic ("architecture-agnostic"), so any concrete architecture (like the FraudNet MLP used here) is an illustrative choice, not a proposal decision.
- **How internal activations become CDAG nodes** — the proposal never specifies the clustering method (k-means, PCA, or otherwise), number of clusters, or which layers get tapped.
- **Exact CDAG node granularity** — one node per raw feature vs. per feature-statistic vs. per feature-group is undefined.
- **GNN's exact input featurization, embedding dimension, and loss function** — only the algorithm family was named.
- **Whether the GNN and surrogate are trained fully jointly (as assumed here) or the GNN is pretrained separately and frozen** — genuinely ambiguous in the proposal.
- **The surrogate's exact training-data generation procedure** — the proposal gestures at "historical retrain outcomes" but never formalizes the offline sandbox-experiment procedure used here to actually produce labeled (node, intervention, outcome) triples.
- **PPO's exact state vector, action set, and reward weights (w1, w2, w3)** — only conceptually described.
- **EWC's exact scope** — which layer(s)/parameters get frozen vs. regularized, how the Fisher information matrix is computed, and replay-buffer design are all unspecified.
- **Shadow/canary promotion thresholds** — what counts as "recovered enough" to promote is not numerically defined.
- **The drift-detection trigger mechanism CADENCE itself uses** — the proposal names PSI/KS-test only as *comparison baselines* to beat, not explicitly as CADENCE's own triggering mechanism; this is a real, unresolved gap worth deciding deliberately.
- **Rolling window sizes** for all the windowed statistics used throughout.
- **The multi-node/multi-layer responsibility threshold** — the mechanics assume a single dominant responsible node; there's no defined rule yet for what happens when responsibility is spread across several nodes/layers at once (see Part 3B).

### What I'd recommend deciding, and why
- **Production model**: keep it architecture-agnostic in the paper, but pick one concrete architecture (an MLP, as used here) for the first working implementation and all initial experiments — trying to stay fully generic from day one will slow down getting anything working.
- **Activation clustering**: k-means on a per-layer basis (as used here), with the number of clusters chosen via a simple elbow-method sweep during development — simple, well-understood, easy to defend in a paper.
- **GNN + surrogate**: train them jointly, end-to-end, as one pipeline (as used here) rather than pretraining the GNN separately — this is simpler to implement first and is a defensible default; pretraining separately could be a later ablation to test if it improves stability.
- **PPO reward weights (w1, w2, w3)**: start with a simple fixed set of weights and tune them via the simulated sandbox environment's overall behavior (does it under- or over-retrain?), rather than trying to derive them analytically — this is standard RL practice and is easy to explain in the paper.
- **Drift trigger**: use PSI + a delayed-label F1 check as CADENCE's own trigger mechanism (not just as a baseline to beat) — it's simple, well-understood, and lets you cleanly report "CDAG's attribution is more precise than PSI *alone*" as a fair comparison, since PSI is still doing the initial alerting, and CDAG's value-add is specifically in the *attribution*, not the detection.
- **Shadow/canary threshold**: define promotion as "new model's F1 on the validation window ≥ the SLA target," which is simple, numeric, and easy to justify.

These are recommendations to unblock building the first version — not settled research decisions. Each is a reasonable, standard default that keeps the project moving, and each is a legitimate thing to revisit or ablate once the first working pipeline exists.

---

# PART III — COMPLETE DATASET REFERENCE (VERIFIED LINKS)

# CADENCE — Complete Dataset List (Verified Links)

Organized by what each dataset validates, since CADENCE is designed to detect drift across *different types* of production models, not just one. All links re-verified via search before this file was written.

---

## 1. Production model datasets (one per model type CADENCE needs to support)

CADENCE's Model Adapter layer needs to be validated against more than one kind of production model to credibly claim "works across model types." Use one dataset per adapter below.

**Neural network classifier adapter**
**Credit Card Fraud Detection Dataset** — 284,807 European credit card transactions, 492 labeled fraud (~0.17%), features `Time`, `Amount`, PCA-transformed `V1`–`V28`.
- **Link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Note:** free Kaggle account required to download. Published by the Machine Learning Group at Université Libre de Bruxelles (ULB).

**Tree ensemble adapter (XGBoost/LightGBM)**
**Give Me Some Credit** — ~150,000 borrower records; predicts probability of serious financial delinquency within two years. A standard benchmark for tree-based credit-risk models.
- **Link:** https://www.kaggle.com/c/GiveMeSomeCredit

**Linear/logistic regression adapter**
**Telco Customer Churn** — 7,043 customer records; predicts churn from account/service/demographic features. Commonly used for interpretable linear/logistic churn models.
- **Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn

**Linear/logistic regression adapter (alternative)**
**UCI Adult (Census Income)** — ~48,842 records; predicts whether income exceeds $50K/year from demographic/employment features. A long-standing standard benchmark for linear classifiers.
- **Link:** https://archive.ics.uci.edu/dataset/2/adult
- **Note:** easiest loaded via the official `ucimlrepo` Python package (`pip install ucimlrepo`), which fetches this exact dataset by ID.

---

## 2. Synthetic drift generators (for controlled, ground-truth-known drift experiments)

**`river`** — Python library with built-in synthetic drift generators (SEA, STAGGER, Hyperplane, RandomRBF, Agrawal).
- **Main site:** https://riverml.xyz
- **Synthetic datasets module docs:** https://riverml.xyz/latest/api/datasets/synth/
- **GitHub:** https://github.com/online-ml/river

**`scikit-multiflow`** — older alternative library with similar generators.
- **Link:** https://scikit-multiflow.github.io/
- **GitHub:** https://github.com/scikit-multiflow/scikit-multiflow

---

## 3. Real-world datasets with known genuine drift

**Electricity (Elec2)** — the standard benchmark dataset in the concept-drift research literature.
- **Link (OpenML):** https://www.openml.org/search?type=data&status=active&id=151
- **Note:** also loadable directly via `river.datasets.Elec2()` without a manual download.

**Airlines delay dataset** — flight delay records with genuine temporal drift.
- **Link (OpenML):** https://www.openml.org/search?type=data&status=active&id=1169

**Criteo Display Advertising dataset** (optional, large-scale)
- **Official source:** https://ailab.criteo.com/ressources/
- **Kaggle (smaller, more manageable subset):** https://www.kaggle.com/c/criteo-display-ad-challenge/data

**Avazu CTR Prediction dataset** (optional, alternative to Criteo)
- **Link:** https://www.kaggle.com/c/avazu-ctr-prediction/data

---

## 4. Continual-learning / forgetting benchmarks

**MNIST** — official source for building Split-MNIST/Rotated-MNIST splits.
- **Official link:** http://yann.lecun.com/exdb/mnist/
- **Note:** also directly downloadable via `torchvision.datasets.MNIST`.

**CIFAR-10** — official source for building Split-CIFAR-10.
- **Official link:** https://www.cs.toronto.edu/~kriz/cifar.html
- **Note:** also directly downloadable via `torchvision.datasets.CIFAR10`.

**Avalanche** — the standard library for building CL benchmark splits from the raw datasets above.
- **Link:** https://avalanche.continualai.org/
- **GitHub:** https://github.com/ContinualAI/avalanche

---

## 5. Text/NLP generality dataset (optional)

**20 Newsgroups** — built directly into `scikit-learn`, no separate download needed.
- **Docs/link:** https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset

**Amazon Reviews (2023 release, with timestamps)** — for topic/vocabulary-shift-over-time experiments.
- **Link:** https://amazon-reviews-2023.github.io/

**Yelp Open Dataset** — alternative for text-drift experiments.
- **Link:** https://www.yelp.com/dataset

---

## 6. Real incident case studies (qualitative, for replayed real-world incidents)

**AI Incident Database** — searchable, maintained database of real AI/ML failure incidents.
- **Link:** https://incidentdatabase.ai/

---

## 7. Carbon/cost estimation (for the RL reward's carbon term)

**Electricity Maps API** — real-time grid carbon-intensity data by region.
- **Link:** https://www.electricitymaps.com/
- **Note:** free developer tier available; requires sign-up for an API key.

**CodeCarbon** — Python package estimating carbon footprint from your own local compute usage.
- **Link:** https://codecarbon.io/
- **GitHub:** https://github.com/mlco2/codecarbon

---

## Quick-reference table

| # | Dataset/Tool | Validates | Link |
|---|---|---|---|
| 1 | Credit Card Fraud Detection | Neural network adapter | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| 2 | Give Me Some Credit | Tree ensemble adapter | https://www.kaggle.com/c/GiveMeSomeCredit |
| 3 | Telco Customer Churn | Linear/logistic adapter | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| 4 | UCI Adult (Census Income) | Linear/logistic adapter (alt.) | https://archive.ics.uci.edu/dataset/2/adult |
| 5 | `river` (synthetic generators) | Ground-truth drift injection | https://riverml.xyz/latest/api/datasets/synth/ |
| 6 | `scikit-multiflow` | Alternative synthetic generators | https://scikit-multiflow.github.io/ |
| 7 | Elec2 | Real known-drift benchmark | https://www.openml.org/search?type=data&status=active&id=151 |
| 8 | Airlines | Real known-drift benchmark | https://www.openml.org/search?type=data&status=active&id=1169 |
| 9 | Criteo Display Ads | Large-scale real CTR data (optional) | https://ailab.criteo.com/ressources/ |
| 10 | Avazu CTR | Large-scale real CTR data (optional) | https://www.kaggle.com/c/avazu-ctr-prediction/data |
| 11 | MNIST | Base data for CL/forgetting tests | http://yann.lecun.com/exdb/mnist/ |
| 12 | CIFAR-10 | Base data for CL/forgetting tests | https://www.cs.toronto.edu/~kriz/cifar.html |
| 13 | Avalanche | Builds CL benchmark splits | https://avalanche.continualai.org/ |
| 14 | 20 Newsgroups | Text-drift generality (optional) | https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset |
| 15 | Amazon Reviews 2023 | Text-drift generality (optional) | https://amazon-reviews-2023.github.io/ |
| 16 | Yelp Open Dataset | Text-drift generality (optional) | https://www.yelp.com/dataset |
| 17 | AI Incident Database | Real incident case studies | https://incidentdatabase.ai/ |
| 18 | Electricity Maps | Carbon-cost estimation | https://www.electricitymaps.com/ |
| 19 | CodeCarbon | Local carbon-footprint estimation | https://codecarbon.io/ |

---

## Where to start, given your hardware (GTX 1650)

Start with **#1 (Credit Card Fraud)** for the neural-network adapter, **#5 (`river`)** for controlled drift injection, and **#7 (Elec2)** for a real-world validation set — all three are small and CPU/GPU-light. Once the core pipeline works end-to-end on the neural adapter, add **#2 (Give Me Some Credit)** to build and prove out the tree-ensemble adapter — this is the single most important "generality" experiment for the paper, since it's what lets you credibly claim CADENCE isn't hardcoded to neural networks. Add **#13 (Avalanche + MNIST)** for the forgetting experiments once that's stable. Save #9/#10 (Criteo/Avazu) and full-size CIFAR-10 runs for free Colab/Kaggle GPU sessions — optional strengthening, not required for the core claims.
