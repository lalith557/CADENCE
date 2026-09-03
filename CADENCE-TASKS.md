# CADENCE — Project Task Checklist

How to use this file: tick a box by changing `- [ ]` to `- [x]`, and fill in the date on the same line. GitHub renders these as clickable checkboxes automatically once pushed.

```
- [ ] Example task not started — Date completed: __________
- [x] Example task finished — Date completed: 2026-08-25
```

This file has two views of the same work:
- **Section A — By Priority**: what matters most if time/energy is limited.
- **Section B — By Sequence**: the actual build order, phase by phase, first task to last.
- **Section C — Additional completed work** not in the original plan (added 2026-08-30).

Keep both updated together — ticking a task in one place doesn't auto-update the other, since this is a plain markdown file, not a linked tracker.

> **Reconciliation note (2026-08-30):** statuses below were reconciled against the actual code, the 17 `R-` entries in `docs/results.md`, and the git history. **"Done" here means implemented and run at least once (smoke scale).** Several validation tasks are marked done at *small-n* (2–5 seeds) and still need ≥10-seed paper-grade re-runs before the paper — those carry an inline note. See `docs/weaknesses.md` (W-1..W-31) and `CADENCE-FINAL-COMPLETION-PROMPT.md` for the remaining work.

---

## SECTION A — Tasks by Priority

### P0 — Critical foundation (nothing else works without these)
- [x] Decide production model architecture for the first build (recommended: MLP, per earlier discussion) — Date completed: 2026-08-26
- [x] Set up project repo structure, virtual environment, and dependency list (PyTorch, PyTorch Geometric, causal-learn, Stable-Baselines3, etc.) — Date completed: 2026-08-26
- [x] Download and verify Credit Card Fraud Detection dataset (neural adapter) — Date completed: 2026-08-24
- [x] Install and test `river` for synthetic drift generation (SEA/STAGGER/Hyperplane) — Date completed: 2026-08-27 *(built a custom drift injector instead; `river` dropped as a hard dep per D-13)*
- [x] Build and train the baseline production model (FraudNet) to a stable F1 — Date completed: 2026-08-27 *(AUROC 0.975, F1 0.798 tuned threshold — R-1)*
- [x] Build the synthetic drift-injection benchmark with known ground-truth cause — Date completed: 2026-08-27
- [x] Decide the drift-detection trigger mechanism (recommended: PSI + delayed-label F1 check) — Date completed: 2026-08-27

### P1 — Core CADENCE pipeline (the actual research contribution)
- [x] Implement activation-hook extraction from the production model's hidden layers — Date completed: 2026-08-29
- [x] Decide and implement the activation-clustering method (recommended: k-means per layer) — Date completed: 2026-08-29
- [x] Implement PC algorithm stage of causal discovery (skeleton construction) — Date completed: 2026-08-29 *(causal-learn PC; falls back to dense skeleton on short windows — known limitation)*
- [x] Implement NOTEARS-style continuous optimization stage (edge weights on PC's skeleton) — Date completed: 2026-08-29
- [x] Build the CDAG data structure (feature nodes + activation-cluster nodes + performance node) — Date completed: 2026-08-29
- [x] Implement the GNN (GraphSAGE/GAT-style) over the CDAG — Date completed: 2026-08-29 *(GCNConv, on GPU)*
- [x] Build the offline sandbox for generating counterfactual training data (inject drift, actually fix it, log outcome) — Date completed: 2026-08-29
- [x] Implement the counterfactual surrogate (MLP or GBM) and train jointly with the GNN — Date completed: 2026-08-29 *(joint train val R²=0.888 — R-Gate-C)*
- [x] Implement responsibility-score calculation (predicted post-fix performance − current performance) — Date completed: 2026-08-29
- [x] Build the Gym-style simulation environment for PPO training (cost/carbon model included) — Date completed: 2026-08-27
- [x] Decide PPO's state vector, action set, and reward weights (w1, w2, w3) — Date completed: 2026-08-29 *(15-d state after Step A enrichment)*
- [x] Train the PPO agent (Retraining Scope Optimizer) in simulation — Date completed: 2026-08-28 *(R-4; retrained on 15-d obs 2026-08-30 — full 10-seed×30k Gate A still pending)*
- [x] Implement EWC (Fisher information computation + penalty term) — Date completed: 2026-08-29
- [x] Implement the partial-retrain executor (layer freezing + EWC fine-tune + replay buffer) — Date completed: 2026-08-29
- [x] Implement the full-retrain executor — Date completed: 2026-08-29
- [x] Decide and implement the multi-node/multi-layer responsibility threshold rule — Date completed: 2026-08-29 *(diffuse-responsibility → full-retrain fallback, §4.4)*
- [x] Implement shadow/canary validation and the promotion/rollback logic — Date completed: 2026-08-29
- [x] Wire the full pipeline together end-to-end (drift → CDAG → GNN → surrogate → PPO → retrain → validate → deploy) — Date completed: 2026-08-29 *(closed loop runs on one command — R-Gate-C)*

### P2 — Validation and generality (what makes it credible, not just working)
- [x] Run end-to-end evaluation on the synthetic benchmark; measure attribution precision/recall vs. ground truth — Date completed: 2026-08-29 *(GNN>PSI, p=0.016; small-n 9 samples — needs 5×10 for paper — R-Gate-B)*
- [x] Validate on Elec2 (real-world known drift) — Date completed: 2026-08-29 *(Pareto option, not strict-dominant; 3 seeds — R-Gate-E)*
- [x] Validate on Airlines delay dataset — Date completed: 2026-08-29 *(hash-encoding masked drift; needs proper encoder — R-Gate-E)*
- [x] Run ablation: remove GNN (raw stats only) — Date completed: 2026-08-29 *(cdag_structural vs gnn_learned in R-Gate-B; not yet a unified ablation table)*
- [x] Run ablation: remove surrogate (naive correlation scoring) — Date completed: 2026-08-29 *(psi_stub baseline in R-Gate-B)*
- [x] Run ablation: replace PPO with a fixed-threshold rule — Date completed: 2026-08-29 *(rule policy used in R-Gate-E/F)*
- [x] Compare against baselines: PSI/KS + full retrain, fixed-schedule retrain, EWC-only continual learning — Date completed: 2026-08-27 *(R-2)*
- [x] Run statistical significance testing across seeds/scenarios — Date completed: 2026-08-29 *(paired Wilcoxon applied throughout; at 2–5 seeds — needs ≥10 for paper)*
- [x] Build the tree-ensemble Model Adapter and validate on Give Me Some Credit — Date completed: 2026-08-29 *(LightGBM; NotImplementedError→full-retrain contract — R-Gate-F)*
- [x] Build the continual-learning forgetting evaluation using Avalanche + MNIST — Date completed: 2026-08-29 *(hand-rolled Split-MNIST loader, not Avalanche; H3 SUPPORTED p=0.031, n=5 — R-Gate-F)*
- [x] Measure compute/carbon savings vs. always-full-retrain baseline — Date completed: 2026-08-29 *(R-3 + R-Gate-E: 69% compute saved on Elec2 vs reactive-full)*

### P3 — Polish, extension, and dissemination (do after P0–P2 are solid)
- [ ] Build the linear/logistic Model Adapter and validate on Telco Churn or UCI Adult — Date completed: __________ *(datasets load in R-0; no dedicated linear adapter validated on them yet — text LogReg adapter exists but on Yelp)*
- [x] Run optional text-drift generality test (20 Newsgroups) — Date completed: 2026-08-30 *(done on Yelp reviews instead; plumbing works, drift not severe — R-Gate-F-text)*
- [ ] Reconstruct 1–2 qualitative case studies from the AI Incident Database — Date completed: __________
- [ ] Write and submit workshop-tier paper draft — Date completed: __________ *(auto-generated skeleton exists in docs/paper/; not yet a written draft)*
- [ ] Prepare patent disclosure draft (Claim A, B, C language) for attorney review — Date completed: __________
- [x] Build public GitHub repo docs, README, architecture diagram — Date completed: 2026-08-27 *(README.md + architecture.md)*
- [x] Set up CI/CD (GitHub Actions: lint, unit tests, benchmark regression) — Date completed: 2026-08-26 *(workflow written + 75 tests green locally; CI never run on a real push — W-8)*
- [x] Dockerize the full pipeline for reproducibility — Date completed: 2026-08-30 *(Dockerfile + docker-compose written & config-validated; stack never booted — W-31)*
- [x] Build a demo dashboard (Grafana or simple React app) — Date completed: 2026-08-30 *(Streamlit; CDAG panel still illustrative, not real NOTEARS weights — W-30)*
- [ ] Write resume bullets and prep interview answers — Date completed: __________
- [ ] Public open-source release (tag v1.0) — Date completed: __________

---

## SECTION B — Tasks in Build Order (Sequence)

Do these roughly in this order — later tasks depend on earlier ones being done first.

### Step 1 — Setup
- [x] 1.1 Set up repo, environment, dependencies — Date completed: 2026-08-26
- [x] 1.2 Decide production model architecture (MLP recommended for v1) — Date completed: 2026-08-26
- [x] 1.3 Download Credit Card Fraud Detection dataset — Date completed: 2026-08-24
- [x] 1.4 Install `river` and confirm synthetic drift generators work — Date completed: 2026-08-27 *(custom injector built; river dropped per D-13)*

### Step 2 — Baseline production model
- [x] 2.1 Build FraudNet architecture — Date completed: 2026-08-27
- [x] 2.2 Train FraudNet on the fraud dataset — Date completed: 2026-08-27
- [x] 2.3 Evaluate baseline F1/AUROC/precision/recall — Date completed: 2026-08-27
- [x] 2.4 Deploy FraudNet in a local "production" scoring loop (simulated live traffic) — Date completed: 2026-08-27

### Step 3 — Drift injection and detection
- [x] 3.1 Build the synthetic drift-injection benchmark with a logged ground-truth cause — Date completed: 2026-08-27
- [x] 3.2 Confirm FraudNet's F1 visibly degrades after injected drift — Date completed: 2026-08-27
- [x] 3.3 Implement the drift-detection trigger (PSI/KS-test on rolling windows) — Date completed: 2026-08-27
- [x] 3.4 Confirm the trigger correctly fires on injected drift and stays quiet otherwise — Date completed: 2026-08-29 *(false-alarm 0.0, true-alert 1.0 — R-Gate-F)*

### Step 4 — Activation collection
- [x] 4.1 Add forward hooks to extract Layer 1 and Layer 2 activations — Date completed: 2026-08-29
- [x] 4.2 Implement per-layer k-means clustering of activations — Date completed: 2026-08-29
- [x] 4.3 Log rolling windowed statistics for each activation cluster — Date completed: 2026-08-29

### Step 5 — CDAG (causal graph)
- [x] 5.1 Define node set (feature nodes + activation-cluster nodes + performance node) — Date completed: 2026-08-29
- [x] 5.2 Implement PC algorithm for skeleton construction — Date completed: 2026-08-29
- [x] 5.3 Implement NOTEARS-style weight fitting restricted to that skeleton — Date completed: 2026-08-29
- [x] 5.4 Validate the CDAG correctly points toward the known injected cause on synthetic data — Date completed: 2026-08-29 *(via GNN readout; structural graph alone weak, SHD-proxy=1.0 — R-Gate-B)*

### Step 6 — GNN + counterfactual surrogate
- [x] 6.1 Build the offline sandbox: inject many synthetic drifts, actually fix each, log (cause, fix, outcome) triples — Date completed: 2026-08-29
- [x] 6.2 Implement the GNN over the CDAG structure — Date completed: 2026-08-29
- [x] 6.3 Implement the surrogate head on top of GNN embeddings — Date completed: 2026-08-29
- [x] 6.4 Train GNN + surrogate jointly on the sandbox dataset — Date completed: 2026-08-29
- [x] 6.5 Validate surrogate's predicted recovery vs. actual recovery (calibration check) — Date completed: 2026-08-29 *(val R²=0.888; over-predicts on unrecoverable drift — a finding — R-Gate-C)*
- [x] 6.6 Implement responsibility-score ranking from surrogate outputs — Date completed: 2026-08-29

### Step 7 — PPO (Retraining Scope Optimizer)
- [x] 7.1 Build the Gym-style simulation environment (drift scenarios + cost/carbon model) — Date completed: 2026-08-27 (Phase 1)
- [x] 7.2 Define state vector, action set, reward function — Date completed: 2026-08-29 (state enriched to 15-d in Step A)
- [x] 7.3 Train PPO agent in simulation — Date completed: 2026-08-28 (R-4); retrained on 15-d obs with Augmented-Lagrangian + EWC-in-sandbox 2026-08-30 (W-28)
- [ ] 7.4 Validate PPO's decisions are sensible (doesn't always pick full retrain, doesn't ignore real drops) — Date completed: __________ *(smoke shows action-mixing, no longer bang-bang; full 10-seed×30k Gate A eval still pending)*

### Step 8 — Retraining execution
- [x] 8.1 Implement no-op path (log-only) — Date completed: 2026-08-29
- [x] 8.2 Implement EWC: Fisher information computation on original training data — Date completed: 2026-08-29
- [x] 8.3 Implement partial-retrain path: layer freezing + EWC-regularized fine-tune + replay buffer — Date completed: 2026-08-29
- [x] 8.4 Implement full-retrain path — Date completed: 2026-08-29
- [x] 8.5 Decide and implement the multi-node responsibility threshold rule — Date completed: 2026-08-29

### Step 9 — Validation and deployment loop
- [x] 9.1 Implement shadow-mode evaluation on held-out recent data — Date completed: 2026-08-29
- [x] 9.2 Implement canary rollout logic — Date completed: 2026-08-29
- [x] 9.3 Implement rollback + failure feedback to the surrogate — Date completed: 2026-08-29
- [x] 9.4 Confirm the full loop runs end-to-end without manual intervention on the synthetic benchmark — Date completed: 2026-08-29

### Step 10 — Real-world and generality validation
- [x] 10.1 Validate full pipeline on Elec2 — Date completed: 2026-08-29 *(3 seeds — needs ≥10)*
- [x] 10.2 Validate full pipeline on Airlines delay dataset — Date completed: 2026-08-29 *(encoder masks drift — rerun needed)*
- [x] 10.3 Build tree-ensemble Model Adapter — Date completed: 2026-08-29
- [x] 10.4 Validate tree-ensemble adapter on Give Me Some Credit — Date completed: 2026-08-29 *(SLA set low; rerun at SLA≈0.45)*
- [x] 10.5 Run continual-learning forgetting tests with Avalanche + MNIST — Date completed: 2026-08-29 *(hand-rolled Split-MNIST; H3 SUPPORTED p=0.031, n=5)*

### Step 11 — Evaluation and comparison
- [x] 11.1 Run all three ablations (no GNN, no surrogate, no PPO) — Date completed: 2026-08-29 *(components run individually; unified ablation table pending)*
- [x] 11.2 Run baseline comparisons (PSI+full retrain, fixed schedule, EWC-only) — Date completed: 2026-08-27
- [x] 11.3 Run statistical significance tests across seeds — Date completed: 2026-08-29 *(at 2–5 seeds; ≥10 needed for paper)*
- [ ] 11.4 Compile final results tables/figures — Date completed: __________ *(auto-skeleton has 13 tables — R-Gate-G; final publication figures not made)*

### Step 12 — Write-up and release
- [ ] 12.1 Draft the research paper (workshop-tier first) — Date completed: __________ *(skeleton auto-generated; narrative unwritten)*
- [ ] 12.2 Draft the patent disclosure (Claims A, B, C) — Date completed: __________
- [x] 12.3 Write repo README, architecture docs, quickstart — Date completed: 2026-08-27
- [x] 12.4 Set up CI/CD and Docker — Date completed: 2026-08-30 *(both configured; CI not run green on a push (W-8), Docker not booted (W-31))*
- [ ] 12.5 Public open-source release — Date completed: __________
- [ ] 12.6 Finalize resume bullets and interview prep — Date completed: __________

---

## SECTION C — Additional completed work (not in the original plan)

Added 2026-08-30. Substantial work that was done but had no task line above.

### GPU / infrastructure
- [x] Add shared device layer (`cadence/common/device.py`); route FraudNet/GNN/surrogate through CUDA; log peak VRAM per run — Date completed: 2026-08-29
- [x] Fix carbon `HardwareProfile` default (gtx-1650 → RTX 4070 Laptop) that biased the RSO reward; add `cadence gpu-check` CLI — Date completed: 2026-08-29
- [x] Test suite: 75 unit + integration tests (incl. 21 fallback tests) — Date completed: 2026-08-30

### RSO / Claim C hardening (Step A)
- [x] Contested-SLA drift scenarios (post-drift F1 just below SLA so no-op is provably wrong) — Date completed: 2026-08-29
- [x] Augmented-Lagrangian dual-λ constraint update for the PPO reward — Date completed: 2026-08-29
- [x] Apply EWC penalty inside the sandbox "partial" action (not just the EWC-only baseline) — Date completed: 2026-08-29
- [x] Enrich PPO observation to 15-d (sla_margin + attribution_concentration) and retrain PPO on it — Date completed: 2026-08-30
- [x] Reward-weight (w_gpu, λ_sla) 3×3 sensitivity sweep — Date completed: 2026-08-28

### Fallback mechanisms (§4)
- [x] Escalation ladder (partial → full → human) — Date completed: 2026-08-29
- [x] Rollback + (predicted vs actual) feedback to the surrogate — Date completed: 2026-08-29
- [x] Diffuse-responsibility guard (spread responsibility → full retrain) — Date completed: 2026-08-29
- [x] Unrecoverable-drift guard (stop burning compute on concept-flip; flag needs-new-labels) — Date completed: 2026-08-29
- [x] False-alarm suppression (PSI fires but delayed-label F1 fine → no-op) — Date completed: 2026-08-29
- [x] Resource/OOM fallback (abort cleanly, keep serving old model) — Date completed: 2026-08-29
- [x] Cold-start / missing-telemetry degrade path — Date completed: 2026-08-29

### Generality & robustness
- [x] Text adapter (TF-IDF + Logistic Regression) + Yelp temporal slice loader — Date completed: 2026-08-30
- [x] Disjoint sub-population split for honest forgetting (H3) measurement — Date completed: 2026-08-29
- [x] Robustness: telemetry-loss (CDAG edge-dropout) AUROC degradation curve — Date completed: 2026-08-29

### Product / docs
- [x] Streamlit dashboard reading committed experiment artifacts — Date completed: 2026-08-30
- [x] docker-compose stack (dashboard + MLflow) + compose config validation — Date completed: 2026-08-30
- [x] Auto paper-skeleton generator (`scripts/build_paper_skeleton.py` → `docs/paper/skeleton.md`, 13 tables) — Date completed: 2026-08-30
- [x] Living docs: `decisions.md`, `tech.md`, `results.md` (17 R-entries), `weaknesses.md` (W-1..W-31), `product.md`, `phase-plan.md` — Date completed: 2026-08-30
- [ ] Prior-art / related-work search (W-4) — Date completed: __________ *(started in `docs/prior_art.md`; full search still open — blocks main-track submission)*

---

## Progress Summary (updated 2026-08-30)

"Done" = implemented and run at least at smoke scale. ⚠️ tasks are done but need ≥10-seed / real-encoder re-runs before the paper (see `docs/weaknesses.md`).

| Section | Total tasks | Completed | % Done |
|---|---|---|---|
| P0 — Critical foundation | 7 | 7 | 100% |
| P1 — Core pipeline | 18 | 18 | 100% |
| P2 — Validation & generality | 11 | 11 (⚠️ small-n) | 100% coded |
| P3 — Polish & dissemination | 11 | 5 | 45% |
| **Section A total** | **47** | **41** | **~87%** |
| Section C — additional work done | 24 | 23 | 96% |

**What remains (the honest gap):** full-scale (≥10-seed) statistical re-runs of H1/H2/H3, the prior-art search (W-4), a written paper draft, the linear adapter on Telco/Adult, AI-Incident case studies, booting Docker + green CI, a real (non-illustrative) dashboard, patent draft, resume/interview prep, and the public v1.0 release. The **coding of the core system is essentially complete**; the remaining work is *evidence at scale*, *productization polish*, and *dissemination*.
