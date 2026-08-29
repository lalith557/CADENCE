# CADENCE — Project Task Checklist

How to use this file: tick a box by changing `- [ ]` to `- [x]`, and fill in the date on the same line. GitHub renders these as clickable checkboxes automatically once pushed.

```
- [ ] Example task not started — Date completed: __________
- [x] Example task finished — Date completed: 2026-08-25
```

This file has two views of the same work:
- **Section A — By Priority**: what matters most if time/energy is limited.
- **Section B — By Sequence**: the actual build order, phase by phase, first task to last.

Keep both updated together — ticking a task in one place doesn't auto-update the other, since this is a plain markdown file, not a linked tracker.

---

## SECTION A — Tasks by Priority

### P0 — Critical foundation (nothing else works without these)
- [ ] Decide production model architecture for the first build (recommended: MLP, per earlier discussion) — Date completed: __________
- [ ] Set up project repo structure, virtual environment, and dependency list (PyTorch, PyTorch Geometric, causal-learn, Stable-Baselines3, etc.) — Date completed: __________
- [ ] Download and verify Credit Card Fraud Detection dataset (neural adapter) — Date completed: __________
- [ ] Install and test `river` for synthetic drift generation (SEA/STAGGER/Hyperplane) — Date completed: __________
- [ ] Build and train the baseline production model (FraudNet) to a stable F1 — Date completed: __________
- [ ] Build the synthetic drift-injection benchmark with known ground-truth cause — Date completed: __________
- [ ] Decide the drift-detection trigger mechanism (recommended: PSI + delayed-label F1 check) — Date completed: __________

### P1 — Core CADENCE pipeline (the actual research contribution)
- [ ] Implement activation-hook extraction from the production model's hidden layers — Date completed: __________
- [ ] Decide and implement the activation-clustering method (recommended: k-means per layer) — Date completed: __________
- [ ] Implement PC algorithm stage of causal discovery (skeleton construction) — Date completed: __________
- [ ] Implement NOTEARS-style continuous optimization stage (edge weights on PC's skeleton) — Date completed: __________
- [ ] Build the CDAG data structure (feature nodes + activation-cluster nodes + performance node) — Date completed: __________
- [ ] Implement the GNN (GraphSAGE/GAT-style) over the CDAG — Date completed: __________
- [ ] Build the offline sandbox for generating counterfactual training data (inject drift, actually fix it, log outcome) — Date completed: __________
- [ ] Implement the counterfactual surrogate (MLP or GBM) and train jointly with the GNN — Date completed: __________
- [ ] Implement responsibility-score calculation (predicted post-fix performance − current performance) — Date completed: __________
- [ ] Build the Gym-style simulation environment for PPO training (cost/carbon model included) — Date completed: __________
- [ ] Decide PPO's state vector, action set, and reward weights (w1, w2, w3) — Date completed: __________
- [ ] Train the PPO agent (Retraining Scope Optimizer) in simulation — Date completed: __________
- [ ] Implement EWC (Fisher information computation + penalty term) — Date completed: __________
- [ ] Implement the partial-retrain executor (layer freezing + EWC fine-tune + replay buffer) — Date completed: __________
- [ ] Implement the full-retrain executor — Date completed: __________
- [ ] Decide and implement the multi-node/multi-layer responsibility threshold rule — Date completed: __________
- [ ] Implement shadow/canary validation and the promotion/rollback logic — Date completed: __________
- [ ] Wire the full pipeline together end-to-end (drift → CDAG → GNN → surrogate → PPO → retrain → validate → deploy) — Date completed: __________

### P2 — Validation and generality (what makes it credible, not just working)
- [ ] Run end-to-end evaluation on the synthetic benchmark; measure attribution precision/recall vs. ground truth — Date completed: __________
- [ ] Validate on Elec2 (real-world known drift) — Date completed: __________
- [ ] Validate on Airlines delay dataset — Date completed: __________
- [ ] Run ablation: remove GNN (raw stats only) — Date completed: __________
- [ ] Run ablation: remove surrogate (naive correlation scoring) — Date completed: __________
- [ ] Run ablation: replace PPO with a fixed-threshold rule — Date completed: __________
- [ ] Compare against baselines: PSI/KS + full retrain, fixed-schedule retrain, EWC-only continual learning — Date completed: __________
- [ ] Run statistical significance testing across seeds/scenarios — Date completed: __________
- [ ] Build the tree-ensemble Model Adapter and validate on Give Me Some Credit — Date completed: __________
- [ ] Build the continual-learning forgetting evaluation using Avalanche + MNIST — Date completed: __________
- [ ] Measure compute/carbon savings vs. always-full-retrain baseline — Date completed: __________

### P3 — Polish, extension, and dissemination (do after P0–P2 are solid)
- [ ] Build the linear/logistic Model Adapter and validate on Telco Churn or UCI Adult — Date completed: __________
- [ ] Run optional text-drift generality test (20 Newsgroups) — Date completed: __________
- [ ] Reconstruct 1–2 qualitative case studies from the AI Incident Database — Date completed: __________
- [ ] Write and submit workshop-tier paper draft — Date completed: __________
- [ ] Prepare patent disclosure draft (Claim A, B, C language) for attorney review — Date completed: __________
- [ ] Build public GitHub repo docs, README, architecture diagram — Date completed: __________
- [ ] Set up CI/CD (GitHub Actions: lint, unit tests, benchmark regression) — Date completed: __________
- [ ] Dockerize the full pipeline for reproducibility — Date completed: __________
- [ ] Build a demo dashboard (Grafana or simple React app) — Date completed: __________
- [ ] Write resume bullets and prep interview answers — Date completed: __________
- [ ] Public open-source release (tag v1.0) — Date completed: __________

---

## SECTION B — Tasks in Build Order (Sequence)

Do these roughly in this order — later tasks depend on earlier ones being done first.

### Step 1 — Setup
- [ ] 1.1 Set up repo, environment, dependencies — Date completed: __________
- [ ] 1.2 Decide production model architecture (MLP recommended for v1) — Date completed: __________
- [x] 1.3 Download Credit Card Fraud Detection dataset — Date completed: 2026-08-24
- [ ] 1.4 Install `river` and confirm synthetic drift generators work — Date completed: __________

### Step 2 — Baseline production model
- [ ] 2.1 Build FraudNet architecture — Date completed: __________
- [ ] 2.2 Train FraudNet on the fraud dataset — Date completed: __________
- [ ] 2.3 Evaluate baseline F1/AUROC/precision/recall — Date completed: __________
- [ ] 2.4 Deploy FraudNet in a local "production" scoring loop (simulated live traffic) — Date completed: __________

### Step 3 — Drift injection and detection
- [ ] 3.1 Build the synthetic drift-injection benchmark with a logged ground-truth cause — Date completed: __________
- [ ] 3.2 Confirm FraudNet's F1 visibly degrades after injected drift — Date completed: __________
- [ ] 3.3 Implement the drift-detection trigger (PSI/KS-test on rolling windows) — Date completed: __________
- [ ] 3.4 Confirm the trigger correctly fires on injected drift and stays quiet otherwise — Date completed: __________

### Step 4 — Activation collection
- [ ] 4.1 Add forward hooks to extract Layer 1 and Layer 2 activations — Date completed: __________
- [ ] 4.2 Implement per-layer k-means clustering of activations — Date completed: __________
- [ ] 4.3 Log rolling windowed statistics for each activation cluster — Date completed: __________

### Step 5 — CDAG (causal graph)
- [ ] 5.1 Define node set (feature nodes + activation-cluster nodes + performance node) — Date completed: __________
- [ ] 5.2 Implement PC algorithm for skeleton construction — Date completed: __________
- [ ] 5.3 Implement NOTEARS-style weight fitting restricted to that skeleton — Date completed: __________
- [ ] 5.4 Validate the CDAG correctly points toward the known injected cause on synthetic data — Date completed: __________

### Step 6 — GNN + counterfactual surrogate
- [ ] 6.1 Build the offline sandbox: inject many synthetic drifts, actually fix each, log (cause, fix, outcome) triples — Date completed: __________
- [ ] 6.2 Implement the GNN over the CDAG structure — Date completed: __________
- [ ] 6.3 Implement the surrogate head on top of GNN embeddings — Date completed: __________
- [ ] 6.4 Train GNN + surrogate jointly on the sandbox dataset — Date completed: __________
- [ ] 6.5 Validate surrogate's predicted recovery vs. actual recovery (calibration check) — Date completed: __________
- [ ] 6.6 Implement responsibility-score ranking from surrogate outputs — Date completed: __________

### Step 7 — PPO (Retraining Scope Optimizer)
- [x] 7.1 Build the Gym-style simulation environment (drift scenarios + cost/carbon model) — Date completed: 2026-08-27 (Phase 1)
- [x] 7.2 Define state vector, action set, reward function — Date completed: 2026-08-29 (state enriched to 15-d in Step A)
- [x] 7.3 Train PPO agent in simulation — Date completed: 2026-08-28 (R-4) / re-run in Step A with Augmented-Lagrangian + EWC-in-sandbox pending
- [ ] 7.4 Validate PPO's decisions are sensible (doesn't always pick full retrain, doesn't ignore real drops) — Date completed: __________ (Gate A eval queued)

### Step 8 — Retraining execution
- [ ] 8.1 Implement no-op path (log-only) — Date completed: __________
- [ ] 8.2 Implement EWC: Fisher information computation on original training data — Date completed: __________
- [ ] 8.3 Implement partial-retrain path: layer freezing + EWC-regularized fine-tune + replay buffer — Date completed: __________
- [ ] 8.4 Implement full-retrain path — Date completed: __________
- [ ] 8.5 Decide and implement the multi-node responsibility threshold rule — Date completed: __________

### Step 9 — Validation and deployment loop
- [ ] 9.1 Implement shadow-mode evaluation on held-out recent data — Date completed: __________
- [ ] 9.2 Implement canary rollout logic — Date completed: __________
- [ ] 9.3 Implement rollback + failure feedback to the surrogate — Date completed: __________
- [ ] 9.4 Confirm the full loop runs end-to-end without manual intervention on the synthetic benchmark — Date completed: __________

### Step 10 — Real-world and generality validation
- [ ] 10.1 Validate full pipeline on Elec2 — Date completed: __________
- [ ] 10.2 Validate full pipeline on Airlines delay dataset — Date completed: __________
- [ ] 10.3 Build tree-ensemble Model Adapter — Date completed: __________
- [ ] 10.4 Validate tree-ensemble adapter on Give Me Some Credit — Date completed: __________
- [ ] 10.5 Run continual-learning forgetting tests with Avalanche + MNIST — Date completed: __________

### Step 11 — Evaluation and comparison
- [ ] 11.1 Run all three ablations (no GNN, no surrogate, no PPO) — Date completed: __________
- [ ] 11.2 Run baseline comparisons (PSI+full retrain, fixed schedule, EWC-only) — Date completed: __________
- [ ] 11.3 Run statistical significance tests across seeds — Date completed: __________
- [ ] 11.4 Compile final results tables/figures — Date completed: __________

### Step 12 — Write-up and release
- [ ] 12.1 Draft the research paper (workshop-tier first) — Date completed: __________
- [ ] 12.2 Draft the patent disclosure (Claims A, B, C) — Date completed: __________
- [ ] 12.3 Write repo README, architecture docs, quickstart — Date completed: __________
- [ ] 12.4 Set up CI/CD and Docker — Date completed: __________
- [ ] 12.5 Public open-source release — Date completed: __________
- [ ] 12.6 Finalize resume bullets and interview prep — Date completed: __________

---

## Progress Summary (update manually as you go)

| Section | Total tasks | Completed | % Done |
|---|---|---|---|
| P0 — Critical foundation | 7 | 0 | 0% |
| P1 — Core pipeline | 18 | 0 | 0% |
| P2 — Validation & generality | 11 | 0 | 0% |
| P3 — Polish & dissemination | 11 | 0 | 0% |
| **Total** | **47** | **0** | **0%** |
