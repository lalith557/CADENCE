# CADENCE — Phase Plan (mirrors CADENCE-BUILD-PROMPT.md §7)

Following the reference's own priority order (§10B: build Claim C first, then A, then B narrowly).

Each phase has a **gate** that must produce a real `R-<n>` entry in `docs/results.md` before the next phase begins. After every phase, run the weakness-audit loop (§6) and log findings to `docs/weaknesses.md`.

---

## Phase 0 — Scaffold ✅ COMPLETE

- Repo skeleton per §15 (`cadence/`, `docs/`, `tests/`, `docker/`, `.github/workflows/`, `configs/`, `experiments/`, `benchmarks/`).
- Config system (`cadence.common.config`) with `default.yaml` (paper), `local.yaml` (dev), `ci.yaml` (smoke).
- Seeds + logging + MLflow wrapper.
- Dataset loaders + CLI (`cadence datasets-check`).
- Test harness + CI (`.github/workflows/ci.yml`).
- Docker stub + Makefile + LICENSE + README + CONTRIBUTING.
- Living docs seeded: decisions.md (D-0..D-12), tech.md, weaknesses.md (W-1..W-7), results.md, product.md, paper/outline.md.

**Gate:** clean venv → `make setup` → `make test` green → CLI `cadence smoke` green. Recorded as R-0.

---

## Phase 1 — FraudNet + drift + trigger + baselines (Claim C foundation) ✅ COMPLETE

**Deliverables:**
- `cadence.adapters.neural.FraudNet` trained on Kaggle Credit Card Fraud, undrifted F1 = 0.798 at tuned threshold. ✅
- `benchmarks.synthetic_drift_gen` — our own drift injector (D-13: dropped river as hard dep), 5 default scenarios, ground truth logged. ✅
- `cadence.collector.drift_trigger` — PSI per-feature + delayed-label F1 (D-10). ✅
- `cadence.rso.env.RetrainingSandboxEnv` — Gym env with FLOPs → GPU-seconds → kg CO₂ closed-form cost model + optional CodeCarbon tracker. ✅
- Baseline runners: (a) PSI+full-retrain, (b) fixed-weekly retrain, (c) EWC-only. ✅

**Gate — R-1..R-3 (2026-08-27):**
- R-1: Pretrained FraudNet AUROC 0.975, F1 (tuned threshold) 0.798 on Credit Card Fraud test set.
- R-2: Baseline strategies × 5 seeds × `amount_multiplicative_abrupt`: PSI+full mean_f1 0.8344 ± 0.0133 (240 μhr GPU), fixed-schedule 0.8229 ± 0.0089 (27 μhr GPU), EWC-only 0.7694 ± 0.0000 (0.86 μhr GPU). Paired Wilcoxon p=0.0625 for the closest pair — borderline, will strengthen at n=10 × 5 scenarios in Phase 2.
- R-3: cost-per-recovery breakdown; PSI+full is the most expensive but most consistent.

Along the way we discovered and fixed:
- W-16: F1@0.5 is meaningless on Fraud → threshold tuning (D-14).
- W-18: harness was applying drift to training data, not held-out slice → fixed.
- W-19: strategy state leaked across seeds (identical F1 fingerprint) → fresh instance per seed.
- W-20: n=5, k=1 is thin → address at Phase 2 with n=10, k=5.
- W-21: negative forgetting = "unrelated" segment is same distribution → redesign at Phase 5.

---

## Phase 2 — RSO / PPO (Claim C core) 🟡 GATE PARTIAL

**Deliverables:**
- `cadence.rso.ppo` — SB3 PPO with fixed-λ Lagrangian reward shaping (D-16). ✅
- Weight sensitivity sweep (W-7). ✅ (script ready; running/queued)
- Training curves logged to MLflow. ✅
- `cadence.rso.scorers.PSIResponsibilityScorer` — stub scorer for Phase 2 (D-16). ✅
- Multi-scenario training env factory (D-15). ✅

**Gate — R-4 (2026-08-28):** measured, honestly reported.
- PPO at D-7 defaults learns a **no-op-forever policy** across 4/5 scenarios — cost drops to 0 at ~0.07 F1 cost. `v14_concept_shift` is unrecoverable for every strategy (fundamental drift-type limitation).
- H2 as originally stated ("cost reduction at equal-or-better F1") is **NOT supported** at default weights. The RSO delivers a controllable trade-off, not a Pareto improvement.
- R-4b (sweep) will test whether other reward-weight cells put RSO on the Pareto frontier.
- W-24: need scenarios where SLA is *contested* (post-drift F1 falls just below SLA) so no-op is definitively suboptimal — before H2 can be "closed."

**R-4b (sensitivity sweep, 3×3 grid) — completed 2026-08-28:**
- All 9 cells produced bang-bang policies (always no-op OR always partial). Zero cells chose full retrain. Zero cells produced a mixed / state-conditional policy.
- Sharp transition boundary around (w_gpu_hr=0.05, λ_sla=1.0). No smooth gradient.
- **No cell delivers a Pareto improvement over baselines.** Best RSO cell (always-partial, F1=0.735) loses to PSI+full (F1=0.834) on F1 by ~0.10, wins on cost by ~350×.
- The paper story that survives: *controllable* cost/F1 trade-off via reward weights — baselines lack this knob.

**Follow-ups queued for Phase 5 re-visit:**
- Augmented-Lagrangian dual updates (W-22, D-16). ✅ landed in Step A (`cadence.rso.lagrangian`).
- EWC penalty inside sandbox partial retrain (W-23). ✅ landed in Step A (`SandboxConfig.ewc_penalty`).
- Longer PPO training (30k+ timesteps) to escape bang-bang. ✅ `phase_a_run.py` defaults to 30k.
- Contested-SLA scenarios (W-24). ✅ `build_contested_sla_scenarios`.
- Continuous / hybrid action space (fraction-of-layers) — new decision needed.

---

## Step A — Claim C rescue (Execution Plan §3) 🟡 CODE COMPLETE, GATE PENDING

Everything Step A of the Execution Plan needs is in-tree; only the long-
running Gate A eval (10 seeds × ≥5 scenarios × 30k PPO steps) hasn't landed
its R-entry yet.

**Landed:**
- `benchmarks.synthetic_drift_gen.build_contested_sla_scenarios` — 3 scenarios
  calibrated so each of {partial, full, no-op} is the correct action somewhere.
  New `MultiFeatureShiftSpec` supports the "needs-full-retrain" case.
- `cadence.rso.env.SandboxConfig` gained `ewc_penalty` / `ewc_fisher_sample_size`
  / `ewc_cache_fisher`; `_do_partial_retrain` now computes Fisher on a historical
  slice, caches theta*, and passes both to `adapter.partial_fit` — the sandbox
  partial now matches Part 3B mechanics (fixes W-23).
- Observation vector widened 13→15 with SLA margin + Herfindahl attribution
  concentration (fixes W-24's "PPO can't tell whether responsibility is
  concentrated or diffuse").
- `cadence.rso.lagrangian.AugmentedLagrangianEnv` — gym.Wrapper that tracks
  per-episode mean SLA violation and updates λ via dual ascent between resets
  (fixes W-22 fixed-λ shortcoming).
- `benchmarks.phase_a_run` — Gate A runner with `--train-timesteps 30000`,
  `--seeds 10`, contested + default scenarios, paired Wilcoxon per scenario.
- Tests: 7 new (4 scenarios, 3 dual-λ), all 45 suite tests green.

**Gate A (H2, contested):** run
```bash
python -m benchmarks.phase_a_run --seeds 10 --train-timesteps 30000 \
    --n-windows 15 --window-size 2048 --sla 0.65
```
Passing means: RSO beats periodic-retrain and reactive-full-retrain on GPU-hr
at equal-or-better F1 recovery, with paired Wilcoxon p<0.05 on contested
scenarios. Failure gets recorded honestly in `results.md` as `R-Gate-A`.

---

## Phase 3 — CDAG (Claim A) 🟡 IN PROGRESS

**Prerequisites:** targeted prior-art search (W-4) still open — deferred until Phase 3 gate lands, since the H1 finding will inform which prior-art angles to search hardest.

**Deliverables:**
- `cadence.cdag.tap` — activation collector with per-layer k-means (D-2). ✅
- `cadence.cdag.nodes` — feature + cluster + performance nodes (D-3). ✅
- `cadence.cdag.discovery` — hybrid PC (causal-learn) + NOTEARS (own numpy impl, D-13 rationale). ✅
- `cadence.attribution.CDAGResponsibilityScorer` — drop-in for `PSIResponsibilityScorer`, Protocol-compatible so RSO doesn't need retraining. ✅ (structural proxy; learned surrogate = Phase 4)
- GNN encoder (D-4) — postponed to Phase 4 (Claim B) since the structural proxy already gives a scorer we can test in R-5.
- Structural Hamming Distance measurement — postponed to R-5b.

**Gate — R-5 (H1):** structural proxy alone LOSES to PSI on hard scenarios (see R-5 / R-Gate-B). Step B replaces the proxy with a real learned GNN and R-Gate-B closes H1 with paired Wilcoxon p<0.05.

---

---

## Step B — Real GNN over CDAG on GPU (Execution Plan §3, Claim A) ✅ CODE COMPLETE + GATE B PASSED (smoke)

**Landed:**
- `cadence.attribution.gnn.CDAGSage` — 2-layer GCNConv over CDAG on cuda, |NOTEARS
  edge weights| via `edge_weight`, 4-dim node features (mean/var/skew/drift_z),
  linear readout per node, self-loop fallback when NOTEARS returns an empty edge
  set.
- `cadence.attribution.sandbox_labels.generate_sandbox_dataset` — offline
  pipeline that produces (CDAG, root-cause-feature) tuples by injecting many
  synthetic drifts (covariate + concept mix), building the CDAG for each, and
  logging the ground truth. Runs the adapter's own predict_proba on GPU.
- `cadence.attribution.gnn_scorer.GNNResponsibilityScorer` — drop-in
  ResponsibilityScorer with the same protocol as PSI/CDAG-structural; RSO
  needs no changes to swap it in.
- `cadence.attribution.gnn_scorer.train_gnn` — supervised cross-entropy
  pretrain on the sandbox tuples, target = root-cause node index. Peak VRAM
  logged to MLflow. All tensors on cuda; refuses silent CPU fallback.
- `benchmarks/phase_b_run.py` — Gate B runner comparing psi vs
  cdag_structural vs gnn_learned, with SHD-proxy measurement on the
  learned CDAG.
- Tests: 5 for GNN (2 @gpu-marked). All 50 project tests green.

**Gate B (R-Gate-B, 2026-08-29):** on the hard-scenario subset (v14_concept_shift,
time_gradual, amount_multiplicative_gradual) × 3 seeds:
- gnn_learned: top1=0.556, MRR=0.633, AUROC=0.927
- psi_stub: top1=0.333, MRR=0.398, AUROC=0.782
- Paired Wilcoxon, gnn > psi on MRR and AUROC: **p=0.0156** (<0.05) ✓
- Paired Wilcoxon, gnn > structural on MRR and AUROC: **p=0.021-0.029** ✓
- GNN peak VRAM during training: 69.8 MB. Wall-clock: 3.0 s at 15 epochs.
- SHD-proxy = 1.0 (learned CDAG under-connects; GNN wins anyway via
  message-passing over cluster nodes) — the openly-recorded weakness.

The full 5-scenario × 10-seed H1 headline for the paper is queued as its own
R-entry; this smoke passes the α=0.05 gate the plan requires.

---

## Phase 4 — Counterfactual surrogate (Claim B, narrow)

**Deliverables:**
- `cadence.attribution.sandbox_labels` — offline pipeline that generates (CDAG, node, intervention, measured post-fix F1) tuples (D-6).
- `cadence.attribution.surrogate` — 3-layer MLP head trained jointly with GNN (D-5).
- Surrogate calibration diagram (predicted vs measured F1).
- Full closed loop: CDAG → GNN → surrogate → responsibility scores → RSO → executor → shadow.

**Gate — R-6:** one command runs the whole loop end-to-end on Credit Card Fraud, MLflow captures every stage.

---

## Phase 5 — EWC executor + shadow/canary

**Deliverables:**
- `cadence.executor.ewc` — Part 3B code from the reference (D-8).
- `cadence.executor.shadow` — shadow window + promotion rule (D-9).
- `cadence.executor.rollback` — outcome feedback to the surrogate.

**Gate — R-7 (H3):** partial retrains show ≤ EWC-only forgetting on unrelated segments, ≥10 seeds, paired Wilcoxon p<0.05.

---

## Phase 6 — Generality + robustness

**Deliverables:**
- `cadence.adapters.tree.LightGBMAdapter` — tree-ensemble adapter (Give Me Some Credit).
- Avalanche+MNIST forgetting benchmark.
- Noisy-telemetry ablation (drop % of activation samples).
- False-alarm-rate measurement.

**Gate — R-8..R-10:** generality (tree adapter beats baselines), robustness curves, false-alarm rate ≤ baselines.

---

## Phase 7 — Productization + paper

**Deliverables:**
- Streamlit dashboard (live drift, CDAG, responsibility, decision, cost/carbon saved).
- Docker-compose (API + dashboard + MLflow).
- `docs/product.md` finalized.
- Paper skeleton (`docs/paper/`) populated from results.md.

**Gate:** clean-clone quickstart works end-to-end; final adversarial audit finds no blocker/major weaknesses.

---

## Weakness-audit loop cadence

Run after every phase gate; run again before declaring the project done. Five lenses: correctness, research-validity, product-readiness, performance/cost, security/privacy. Rank each finding as blocker/major/minor × low/med/high effort. Fix blockers+majors before the next phase.
