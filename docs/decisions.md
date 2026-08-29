# CADENCE — Decision Log

Every real design decision lives here. Format per entry:

```
### D-<n>: <title>   (<YYYY-MM-DD>)
- **Context:** what forced the decision (link to Part 6 gap if applicable)
- **Decision:** what was chosen, concretely
- **Why:** the reasoning
- **Alternatives considered:** and why rejected
- **Reversibility:** cheap / costly, and what would make us revisit
- **Affects:** claim (A/B/C), metric, module
```

---

### D-0: Adopt Part 6's recommended defaults as Phase-0 starting points   (2026-08-26)
- **Context:** The master reference's Part 6 "Honest Gap Analysis" enumerates ~12 design gaps left open by the original proposal and offers a recommended default for each. Rather than re-litigate all of them before writing code, we adopt the Part 6 recommendations as our Phase-0 starting configuration and record each as its own decision below (D-1..D-10). Each remains ablatable.
- **Decision:** Take the Part 6 recommendations as-is for Phase 0. Any deviation later requires its own D-<n> entry.
- **Why:** Ships the first working pipeline fastest, and the recommendations are already the standard, defensible choices that a reviewer would expect. All are cheap to swap.
- **Alternatives considered:** Do a full literature scan of each gap first — rejected; would delay a working baseline by weeks with no gain in credibility, since these defaults are themselves the standard choices.
- **Reversibility:** Cheap. Each downstream decision is scoped to a single module.
- **Affects:** All claims and modules.

---

### D-1: Production model architecture = FraudNet MLP for the neural adapter   (2026-08-26)
- **Context:** Part 6 gap — "Production model architecture: deliberately left generic." Need one concrete architecture to instantiate the neural-network adapter.
- **Decision:** 30 → Dense(64, ReLU, dropout 0.2) → Dense(32, ReLU, dropout 0.2) → Dense(1, Sigmoid), exactly the FraudNet spec in Part 3 Table A. Trained with class-weighted BCE, Adam @ 1e-3, batch 256, up to 30 epochs with early stopping.
- **Why:** Matches the master reference's running example verbatim, keeps the walkthrough numbers as an anchor for eyeballing pipeline correctness, and is tiny enough to iterate on CPU let alone the RTX 4070 (see D-11).
- **Alternatives considered:** (a) A ResNet-style tabular arch (TabResNet) — rejected: overkill for 30-feature tabular, hides drift signal in extra depth; (b) A wider MLP — rejected: not what the reference specifies, would drift from the anchor.
- **Reversibility:** Cheap — hidden behind the ModelAdapter interface.
- **Affects:** Claim A/B/C via CDAG node-count; the neural adapter tests.

### D-2: Activation clustering = per-layer k-means, k chosen by elbow method   (2026-08-26)
- **Context:** Part 6 gap — "how internal activations become CDAG nodes" was unspecified.
- **Decision:** For each tapped layer, fit k-means on the training-set post-ReLU activation vectors. Choose k by silhouette-score sweep over {2..min(16, layer_width/4)} on a held-out slice, with elbow as tiebreaker. For FraudNet defaults: Layer 1 → 8 clusters, Layer 2 → 4 clusters (the walkthrough numbers), overridden by the sweep if it clearly disagrees.
- **Why:** Standard, well-understood, defensible in a paper. k-means centroids give a natural "cluster centroid distance-from-baseline" statistic per node per window.
- **Alternatives considered:** (a) PCA components as nodes — rejected: components aren't as human-interpretable as "this cluster of neurons fires on high-Amount transactions"; (b) Individual neurons as nodes — rejected: blows up CDAG node count and hurts causal-discovery sample efficiency.
- **Reversibility:** Cheap — clustering swap is isolated to cadence/cdag/nodes.py.
- **Affects:** Claim A (CDAG quality), attribution precision.

### D-3: CDAG node granularity = one node per external feature + one node per activation cluster + one performance node   (2026-08-26)
- **Context:** Part 6 gap — "one node per raw feature vs. per feature-statistic vs. per feature-group is undefined."
- **Decision:** One CDAG node per raw external feature (windowed stats become that node's feature vector), one node per activation cluster from D-2, one node for windowed performance metric. For FraudNet: 30 + 8 + 4 + 1 = 43 nodes.
- **Why:** Simplest mapping; matches how the walkthrough numbers are laid out; keeps node count small enough for PC/NOTEARS to fit reliably from short windows.
- **Alternatives considered:** (a) Per-feature-statistic node (mean, var, skew are separate nodes) — rejected: quadruples node count and edges become less interpretable ("var_of_Amount → cluster3" is a mouthful); (b) Feature-group nodes — rejected: requires hand-labeled groupings.
- **Reversibility:** Cheap — node builder is a single module.
- **Affects:** Claim A, attribution granularity.

### D-4: GNN = 2-layer GraphSAGE, hidden 32, output embedding 16, mean aggregator   (2026-08-26)
- **Context:** Part 6 gap — "GNN's exact input featurization, embedding dimension, and loss function."
- **Decision:** 2 message-passing layers, hidden 32, output 16-dim node embeddings, mean aggregator, ReLU, dropout 0.1. Input node features = windowed (mean, var, skew, drift_score, distance_from_baseline). Edge attribute = NOTEARS weight from CDAG.
- **Why:** 2 layers is enough for the 2-hop feature→cluster→performance paths that dominate CDAG; small dims keep GNN training cheap; GraphSAGE handles the variable node population when new activation clusters emerge.
- **Alternatives considered:** GAT (attention) — rejected as ablation candidate, not default; GCN — rejected: needs symmetric normalization we don't want for directed causal edges.
- **Reversibility:** Cheap — swappable via config.
- **Affects:** Claim A, attribution accuracy.

### D-5: GNN + surrogate trained jointly, end-to-end   (2026-08-26)
- **Context:** Part 6 gap — "whether GNN and surrogate are trained jointly or GNN is pretrained separately."
- **Decision:** Joint end-to-end training with a single MSE loss on measured post-intervention F1. Surrogate = 3-layer MLP head (32 → 32 → 1) over concatenation of (GNN node embedding, context vector).
- **Why:** Simplest defensible starting point; GNN's embeddings get shaped directly by the downstream regression signal that matters.
- **Alternatives considered:** Two-stage (pretrain GNN with a self-supervised graph objective, then freeze) — kept as a planned ablation.
- **Reversibility:** Cheap.
- **Affects:** Claim A + B (attribution calibration).

### D-6: Surrogate offline training data = synthetic sandbox drift + logged historical retrains   (2026-08-26)
- **Context:** Part 6 gap — "surrogate's exact training-data generation procedure."
- **Decision:** Offline pipeline generates (CDAG, candidate_node, intervention_type, measured_post_fix_F1) tuples by (a) injecting a `river` drift with a known root-cause node, (b) actually running the corresponding partial/full retrain in-sandbox, (c) measuring resulting F1. Target ~5k tuples per production-model type before first RSO episode.
- **Why:** This is the only way to get labeled counterfactual-outcome data without running experiments against live traffic.
- **Alternatives considered:** Use only historical retrain outcomes — rejected: too few and too biased toward the historical decisions engineers happened to make.
- **Reversibility:** Costly — the tuple format bakes into surrogate + GNN training; changes require regeneration.
- **Affects:** Claim B primarily.

### D-7: PPO state = top-k=5 responsibility scores + drift duration + current F1 + SLA target + per-action GPU-hr estimates + current grid gCO2/kWh + hours-since-last-retrain; actions = {no-op, partial, full}; reward = ΔF1 − w1·GPU_hr − w2·kg_CO2 − λ·max(0, SLA − new_F1)   (2026-08-26)
- **Context:** Part 6 gap — "PPO's exact state vector, action set, and reward weights."
- **Decision:** State dim = 5 + 1 + 1 + 1 + 3 + 1 + 1 = 13. Action space = Discrete(3). Weights w1=0.05 (per-GPU-hr penalty in F1-units), w2=0.5 (per-kg-CO2 penalty), λ (Lagrangian multiplier on SLA violation) initialized 1.0 and updated by the augmented-Lagrangian scheme every 10 policy updates.
- **Why:** Compact state that captures every decision-relevant signal without being sparse. Discrete-3 matches the reference's original action set. Weights are hyperparams to tune in-sandbox.
- **Alternatives considered:** Continuous action (fraction of layers to unfreeze) — rejected: harder to explain and harder to tie back to Part 3B mechanics.
- **Reversibility:** Moderate — state changes force policy retraining. Weight changes are cheap.
- **Affects:** Claim C.

### D-8: EWC scope = only the layer(s) mapped from responsible nodes; Fisher from a 5k-sample of original training data; replay buffer ratio 70% new / 30% replayed   (2026-08-26)
- **Context:** Part 6 gap — "EWC's exact scope."
- **Decision:** Freeze all non-target layers via requires_grad=False. Compute diagonal Fisher over a fixed 5k-sample of the pre-drift training set, cached per model. Fine-tune LR = 1e-4, up to 10 epochs, early stop on validation F1. EWC penalty λ_ewc starts at 1000, tuned per model.
- **Why:** Direct instantiation of the Part 3B code sketch; keeps memory footprint bounded and Fisher stable.
- **Alternatives considered:** Online EWC (running Fisher) — kept as future work; full-parameter EWC without freezing — rejected: defeats the "partial" cost story.
- **Reversibility:** Cheap.
- **Affects:** Claim C forgetting metric.

### D-9: Shadow/canary promotion threshold = new-model F1 ≥ SLA target on a 24h shadow window   (2026-08-26)
- **Context:** Part 6 gap — "shadow/canary promotion thresholds."
- **Decision:** Promote iff shadow-window F1 ≥ SLA target AND shadow-window F1 ≥ pre-drift F1 − ε (ε = 0.02) on unrelated held-out segments (guards against forgetting-induced regressions elsewhere).
- **Why:** Cheap, numeric, and directly checkable. The unrelated-segment guard is what turns H3 into a real deployment-time invariant, not just an offline metric.
- **Alternatives considered:** Sequential probability ratio test — rejected as over-engineering for Phase 0; kept as future improvement.
- **Reversibility:** Cheap.
- **Affects:** Claim C safety.

### D-10: CADENCE's own drift trigger = PSI on features + delayed-label rolling F1   (2026-08-26)
- **Context:** Part 6 gap — "the drift-detection trigger mechanism CADENCE itself uses."
- **Decision:** PSI computed per feature per 1h rolling window; alert fires when any feature's PSI > 0.25 OR when rolling F1 over the last 24h of delayed-label traffic drops below SLA. Baselines (PSI+full-retrain, fixed-weekly, EWC-only) share the same trigger for a fair comparison; CDAG only ever competes on *attribution*, never on detection.
- **Why:** Keeps the comparison honest — CDAG's value-add is attribution, not detection, so it must not gain from using a better trigger than baselines. PSI+F1 is the standard trigger the reference proposes.
- **Alternatives considered:** ADWIN / DDM detectors — kept as ablation.
- **Reversibility:** Cheap.
- **Affects:** All claims — fairness of the experimental comparison.

### D-11: Target Python = 3.10; target GPU = the machine's actual RTX 4070 (8 GB) rather than the assumed GTX 1650 (4 GB)   (2026-08-26)
- **Context:** The brief assumed a GTX 1650 with 4 GB. `nvidia-smi` on the dev machine reports an RTX 4070 (8 GB). Python 3.10 is available alongside the default Python 3.13.
- **Decision:** Use Python 3.10 (broadest wheel availability for PyTorch Geometric, Stable-Baselines3, causal-learn). Design the default pipeline to fit within 4 GB VRAM so it reproduces on the reference-spec GTX 1650, but allow optional configs that use the extra 4 GB on this machine for larger sandboxes / more parallel seeds.
- **Why:** Keeping the default at GTX-1650 budget preserves the paper's reproducibility claim; using the extra headroom locally speeds iteration without changing the reported numbers.
- **Alternatives considered:** Python 3.13 (default installed) — rejected: torch-geometric wheels lag Python releases.
- **Reversibility:** Costly — bumping Python later would rebuild the pinned lockfile.
- **Affects:** All modules; the CI matrix pins 3.10.

### D-17: Narrow Claim B to a strict dependent of Claim A, in response to prior-art findings   (2026-08-28)
- **Context:** First-pass prior-art search (W-4) found GNN+causal-graph uplift papers (arxiv 2311.08434, 2403.06489) and long-term latent-surrogate work (ScienceDirect 2024) that share Claim B's technical construction (graph node → treatment-effect estimate via learned surrogate) even though their applications differ (marketing uplift, long-horizon RCTs). Broadly-worded Claim B would collide.
- **Decision:** Rewrite Claim B as a dependent claim of A, requiring ALL of:
  1. Graph nodes span external features AND internal activation clusters (from Claim A).
  2. Surrogate training data = `(CDAG-node, intervention-type, MEASURED post-fix F1)` triples from sandbox retrain experiments — NOT observational marketing data.
  3. Intervention set = `{no-op, partial-subnetwork-retrain, full-retrain}` — the RSO action space from Claim C.
- **Why:** Keeps Claim B defensible after prior-art review. The combination of (activation-cluster nodes × retrain-outcome training signal × retrain-scope action set) is the still-defensible narrow contribution.
- **Alternatives considered:** Drop Claim B entirely and rely on A+C only. Rejected: the surrogate is what lets us score responsibility without running actual retrains — a real technical contribution worth naming, just narrowly.
- **Reversibility:** Cheap — only affects paper method-section language and patent claim wording, not code.
- **Affects:** Patent claim breadth; paper's related-work framing.

### D-16: Lagrangian SLA penalty in RSO reward starts as fixed-λ shaping, not dual-update   (2026-08-27)
- **Context:** D-7 called for an "augmented-Lagrangian" scheme where λ is updated every 10 policy updates based on running SLA-violation rate. That's the *right* long-run approach but adds tuning + wiring complexity we don't want to eat during Phase 2. The reference brief acknowledges the reward-shaping variant as a legitimate starting point.
- **Decision:** Phase 2 uses fixed λ=1.0 (D-7 initial value) baked into the reward. The reward = ΔF1 − w1·GPU_hr − w2·kg_CO2 − λ·max(0, SLA − post_F1). If Phase 2's sensitivity sweep shows the policy is too permissive (violates SLA often) or too aggressive (over-retrains), Phase 5 will introduce dual updates.
- **Why:** Ships a real, measurable H2 test faster. Also — fixed λ makes the sensitivity sweep interpretable (you sweep λ directly instead of watching a self-tuning multiplier chase a moving target).
- **Alternatives considered:** Full dual-updating Lagrangian from the start (postponed to Phase 5); constrained-PPO (CPO / PPO-Lagrangian variants — not in SB3 core, would need to add a Safety-Gymnasium dep).
- **Reversibility:** Cheap — swap the reward shaping for a dual-update on the multiplier.
- **Affects:** Claim C.

### D-15: PPO trains across all eval scenarios round-robin (multi-scenario factory)   (2026-08-27)
- **Context:** Initial smoke run trained PPO on a single scenario (`amount_multiplicative_abrupt`). The learned policy overfit to that drift and generalized poorly to the other four scenarios at eval time.
- **Decision:** `build_multi_scenario_env_factory` in `cadence.rso.ppo` cycles the training env through pre-injected copies of every scenario, so PPO sees ~1/N of episodes per scenario. This is standard "domain randomization" for RL generalization.
- **Why:** The paper's H2 claim requires RSO to beat baselines on scenarios it hasn't specifically trained on, or on all scenarios simultaneously. Overfitting to one drift shape doesn't test the interesting question.
- **Alternatives considered:** (a) Train N separate policies, one per scenario. Rejected: reviewers rightly ask "what policy would you deploy?" — the answer must be "one." (b) Learn a scenario-conditioned policy. Rejected: needs a scenario embedding on the state vector we don't have yet.
- **Reversibility:** Cheap.
- **Affects:** Claim C generalization story.

### D-14: Tune the neural adapter's decision threshold on validation, per fit and per partial_fit   (2026-08-27)
- **Context:** Credit Card Fraud is 0.17% positive. FraudNet trained with class-weighted BCE hits AUROC ≈ 0.975 on the undrifted test set (excellent ranking), but F1 at the naïve 0.5 threshold collapses to ≈ 0.18 because the model is calibrated toward high recall (0.90) at very low precision (0.10). Every downstream number (baseline F1, post-retrain F1, forgetting) was measured at 0.5 in the first Phase 1 dry-run and looked artificially bad.
- **Decision:** After every `fit` and `partial_fit`, run a 101-point threshold sweep on the validation set (or on the fine-tune data if no val is passed) and pick the threshold that maximizes F1. Store as `adapter.decision_threshold`. All F1 measurements (harness `_f1_of`, sandbox `_eval_f1`, PSI-trigger's delayed-label F1) use this attribute rather than a hard-coded 0.5.
- **Why:** Standard practice for extreme imbalance. Reporting F1@0.5 on this dataset would misrepresent the model's actual utility and would systematically make ALL retraining strategies look identical (they all sit at the same bad threshold). The threshold is a per-model artifact, not a global hyperparameter, so re-tuning after each partial retrain is essential — the post-partial-fit distribution shifts.
- **Alternatives considered:** (a) Report AUROC/AUPRC only. Rejected: the reference proposal names F1 as the primary metric for FraudNet and the RSO's SLA is stated in F1 terms. (b) Use a fixed 0.05 threshold (~pos-rate). Rejected: fine on Fraud, doesn't generalize to GMSC / Telco / Adult which have different base rates. (c) Report both F1@0.5 and F1@tuned. Kept as a supplementary table in the paper, but the RSO reward + SLA use tuned-F1.
- **Reversibility:** Cheap. Threshold sweep is one method on the adapter.
- **Affects:** Every measured F1 number in results.md; the SLA definition; H1/H2/H3 comparisons.

### D-13: Drop `river` as a hard dep; use our own `DriftInjector` and treat Elec2 as optional   (2026-08-27)
- **Context:** `pip install river` fails on Windows without a Rust toolchain (river 0.22+ has Rust-backed extensions and only ships sdists for most Python/OS combos). We hit this concretely today: build fails with "can't find Rust compiler."
- **Decision:** Move `river` from the `[drift]` extra to an *optional* extra (`[river]`) that users can install if they want the Elec2 loader or the river-native drift generators. Our own `benchmarks/synthetic_drift_gen/injector.py` covers the drift generation Phase 1 needs (feature shift + concept shift, gradual + abrupt, ground truth logged). The `load_elec2` loader gracefully raises `ImportError` if river isn't installed.
- **Why:** Unblocks Phase 1 immediately without asking a reviewer to install a Rust toolchain. Our injector is more useful anyway because it emits *ground truth* alongside the drift — river's built-in generators do not.
- **Alternatives considered:** (a) Install Rust and pin `river>=0.21`. Rejected: adds a heavy system-level dep the paper's Docker image would also need to build, hurting reproducibility. (b) Use `scikit-multiflow`. Rejected: unmaintained since 2021, no Elec2 loader helper, and depends on scikit-learn <1.0 which conflicts with our current install.
- **Reversibility:** Cheap — the `[river]` extra is a one-line install. Loader stays the same.
- **Affects:** No claim; only convenience of Elec2 loading. All three hypotheses can still use synthetic drift from our injector for Phase 1-4 without touching Elec2.

### D-12: Multi-node / diffuse-responsibility fallback = unfreeze every layer whose top-mapped node has responsibility ≥ 0.15, and if that set covers ≥60% of parameters, escalate to full retrain   (2026-08-26)
- **Context:** Part 3B "real gap worth flagging" + Part 6 — the single-dominant-node assumption breaks when drift is diffuse.
- **Decision:** τ_partial = 0.15 (unfreeze threshold), τ_diffuse = 0.60 (fraction of trainable params triggering escalation). Both configurable per model.
- **Why:** Gives a principled rule for the multi-node case without expanding PPO's action space; the escalation rule matches what an engineer would intuitively do.
- **Alternatives considered:** Add "multi-partial" as a fourth PPO action — rejected: expands action space and requires more training data.
- **Reversibility:** Cheap.
- **Affects:** Claim C partial-retrain safety.
