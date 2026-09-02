# CADENCE — Weaknesses Register

Living self-audit. Every finding gets logged; blockers/majors get fixed and re-verified; minors get a plan.

Format per entry:
```
### W-<n>: <title>   (<YYYY-MM-DD>)
- **Lens:** correctness / research-validity / product-readiness / performance-cost / security-privacy
- **Finding:** what's wrong or worrying
- **Severity × Effort:** blocker/major/minor × low/med/high
- **Fix plan / owner:** what to do
- **Status:** open / in-progress / fixed (with commit or R-<n> for verification)
```

---

## Phase 0 seed findings (self-audit before writing any pipeline code)

### W-1: Reference brief assumed GTX 1650 (4 GB); actual dev machine is RTX 4070 (8 GB)   (2026-08-26)
- **Lens:** correctness / research-validity
- **Finding:** The build prompt and Part III's "Where to start" both assume 4 GB VRAM. Local `nvidia-smi` shows 8 GB. If we silently use the extra memory, our "runs on GTX 1650" reproducibility claim becomes false.
- **Severity × Effort:** major × low
- **Fix plan:** Track two configs in `configs/`: `default.yaml` sized for 4 GB (what the paper reports), `local.yaml` allowed to use 8 GB for faster iteration only. Document in README + tech.md. Never let a `local.yaml` run populate `results.md`.
- **Status:** open — plan documented in D-11; enforcement lives in the config loader.

### W-2: Kaggle datasets committed as raw CSVs risk repository bloat and licensing violation   (2026-08-26)
- **Lens:** product-readiness / security-privacy
- **Finding:** `Dataset/` contains ~285k-row CSVs downloaded from Kaggle. Committing them (a) violates Kaggle's redistribution terms for some datasets, (b) makes clone slow.
- **Severity × Effort:** major × low
- **Fix plan:** Add `Dataset/` to `.gitignore`. Loaders assert file exists with a helpful error pointing to the Kaggle URL. Provide `scripts/setup_datasets.py` that verifies file presence and hashes.
- **Status:** open — will fix in Task #10 (.gitignore).

### W-3: CIFAR-10 is present but only as .7z archives; MNIST is present as ubyte binaries   (2026-08-26)
- **Lens:** correctness
- **Finding:** `Dataset/cifar-10/{train.7z, test.7z}` — no extracted images. `Dataset/MNIST/*.ubyte` present but not in a `torchvision.datasets.MNIST` folder structure.
- **Severity × Effort:** minor × low
- **Fix plan:** Loaders should let torchvision download to a project-local `~/.cadence/torchvision_cache/` rather than trying to parse the raw `Dataset/MNIST/` blobs. CIFAR-10 is optional and Colab-only per the brief.
- **Status:** open — will fix in Task #6 (loaders).

### W-4: No prior-art search yet for Claims A, B, C   (2026-08-26)
- **Lens:** research-validity
- **Finding:** The novelty claims in Section 10B name candidate prior-art areas but no actual search has been done. Doing this before we invest in the surrogate (Claim B, highest risk) is the cheap version.
- **Severity × Effort:** major × medium
- **Fix plan:** Do a targeted Google Scholar / Semantic Scholar sweep in Phase 3 (before building the CDAG). Log findings in `docs/prior_art.md`. If any hit is too close, narrow the claim before more building.
- **Status:** open — deferred to Phase 3 kickoff; do not build Claim A code before this pass.

### W-5: "5 seeds" is thin for the paper's H1/H2/H3 tests   (2026-08-26)
- **Lens:** research-validity
- **Finding:** The brief says ≥5 seeds, but for effect sizes with high variance (RL rewards especially) 5 seeds gives loose CIs. Reviewers routinely ask for 10.
- **Severity × Effort:** minor × low (extra compute is cheap here)
- **Fix plan:** Default to 10 seeds for the headline runs; keep 5 for ablations to save compute.
- **Status:** open — bake into the experiment configs.

### W-6: We haven't defined "SLA target" for FraudNet numerically anywhere shippable   (2026-08-26)
- **Lens:** correctness / research-validity
- **Finding:** The walkthrough uses F1 SLA = 0.90, but that's an illustrative number for a walkthrough, not a defensible production target. The paper needs to say why the SLA is set where it is per dataset.
- **Severity × Effort:** minor × low
- **Fix plan:** Set SLA = 0.95 × baseline-F1 for each dataset (i.e., allow at most a 5% relative F1 drop before triggering). Rationale is defensible in a paper's method section.
- **Status:** open — add as D-13 when the dataset baselines are measured (Phase 1).

### W-7: PPO reward weights (w1, w2, λ) are guessed, not calibrated   (2026-08-26)
- **Lens:** research-validity
- **Finding:** D-7 pins w1=0.05, w2=0.5, λ_init=1.0. These are guesses. A hostile reviewer will ask how they were chosen.
- **Severity × Effort:** major × medium
- **Fix plan:** In Phase 2, run a sensitivity sweep over each weight and report the working range in `results.md`. Choose values from the center of the stable range.
- **Status:** open — must be resolved before H2 numbers land.

---

## Phase 0 post-gate audit (2026-08-26)

Audited across all five lenses after the Phase 0 gate passed (R-0).

### W-8: CI workflow has never actually run in GitHub Actions   (2026-08-26)
- **Lens:** product-readiness
- **Finding:** `.github/workflows/ci.yml` exists and looks correct, but the repo isn't pushed to GitHub yet — no run has verified that Ubuntu-runner pip resolution actually works or that the CLI smoke passes in a container without user site-packages.
- **Severity × Effort:** major × low
- **Fix plan:** Push to a GitHub repo and confirm the CI badge is green. Add a job that also builds `docker/Dockerfile` to catch container-vs-Windows drift.
- **Status:** open — do this before Phase 1 lands.

### W-9: Docker image isn't built or exercised   (2026-08-26)
- **Lens:** product-readiness / research-validity
- **Finding:** `docker/Dockerfile` written, never `docker build`ed. The paper's reproducibility appendix demands a working image.
- **Severity × Effort:** major × low
- **Fix plan:** Build locally, verify `docker run cadence cadence --help` works, then add a CI job that builds it on every PR to `main`. Attach the image to GitHub Container Registry once Phase 7 begins.
- **Status:** open.

### W-10: 16 upstream Pydantic-v1 deprecation warnings from MLflow clutter test output   (2026-08-26)
- **Lens:** correctness (cosmetic)
- **Finding:** MLflow 2.19's `gateway/config.py` uses old `@validator` decorators; every pytest run prints 16 warnings. Fine functionally, noisy for a reviewer skimming the log.
- **Severity × Effort:** minor × low
- **Fix plan:** Either pin `mlflow>=2.21` once its tracking-server API stabilizes, or add a `filterwarnings` entry in `pyproject.toml [tool.pytest.ini_options]` scoped to `mlflow.gateway`.
- **Status:** open — will filter in a follow-up commit.

### W-11: mypy is `continue-on-error: true` in CI (advisory only)   (2026-08-26)
- **Lens:** research-validity / product-readiness
- **Finding:** Type errors don't fail CI in Phase 0. Fine while modules are stubs; must tighten as real code lands.
- **Severity × Effort:** minor × low
- **Fix plan:** Remove `continue-on-error` at the Phase 2 gate. Add py.typed marker and start adding annotations on module boundaries.
- **Status:** open — Phase 2.

### W-12: No pre-commit hook — ruff violations sneak in until CI catches them   (2026-08-26)
- **Lens:** product-readiness (dev experience)
- **Finding:** Small QoL gap; a `pre-commit` config that runs ruff+ruff-format+mypy on staged files would fail fast locally.
- **Severity × Effort:** minor × low
- **Fix plan:** Add `.pre-commit-config.yaml` in Phase 1 alongside the FraudNet code.
- **Status:** open.

### W-13: Three near-identical config YAMLs (default/local/ci) — will drift when new options land   (2026-08-26)
- **Lens:** product-readiness (maintenance)
- **Finding:** The three profiles duplicate ~40 lines of config each. Adding a new field means editing three places.
- **Severity × Effort:** minor × medium
- **Fix plan:** Introduce a `configs/base.yaml` and let profile YAMLs be pure overrides via pydantic-settings' merge, OR generate the profile YAMLs from the pydantic defaults with `cadence config-dump`. Not worth doing until the config surface stabilizes (Phase 2+).
- **Status:** open — accept the duplication for now; revisit when it bites.

### W-14: `mlflow.log_dict(config.json)` writes an artifact per run — grows unbounded   (2026-08-26)
- **Lens:** performance-cost
- **Finding:** Every run duplicates ~2 KB of config as an MLflow artifact. Negligible for now; would grow with thousands of experiments.
- **Severity × Effort:** minor × low
- **Fix plan:** After Phase 5, garbage-collect old MLflow artifacts and store config hashes only. Include a `cadence gc-experiments` command.
- **Status:** open — deferred.

### W-15: No secrets story documented — Electricity Maps API key is called out in the brief   (2026-08-26)
- **Lens:** security-privacy
- **Finding:** The reference brief calls Electricity Maps API key "via env var" but there's no `docs/secrets.md`, no `.env.example`, no doc on where the key goes. CodeCarbon is the offline fallback, so we can defer this, but it's a real omission.
- **Severity × Effort:** minor × low
- **Fix plan:** Ship `.env.example` and `docs/secrets.md` in Phase 1 alongside the carbon module.
- **Status:** open — Phase 1.

**Phase 0 audit summary:** 8 new items logged (W-8..W-15). One major (W-8, CI never run) plus one major (W-9, Docker never built) — both low-effort; scheduled to fix before Phase 1 gate. No blockers. Register grew but every entry has a plan.

---

## Phase 1 mid-build findings (2026-08-27)

### W-16: F1 at the naïve 0.5 threshold is meaningless on Credit Card Fraud   (2026-08-27)
- **Lens:** research-validity / correctness
- **Finding:** The first Phase 1 dry-run reported baseline mean_f1 ≈ 0.25 across all three strategies. Investigation shows the pretrained FraudNet has AUROC = 0.975 but F1 = 0.185 at threshold 0.5, because the class-weighted BCE loss pushes recall to 0.90 while precision sits at 0.10. All three baselines were being evaluated at the same bad threshold, making them look artificially identical.
- **Severity × Effort:** blocker × low
- **Fix plan:** Tune the decision threshold on validation and store it on the adapter (D-14). Re-run Phase 1 gate.
- **Status:** fixed — see D-14, commit forthcoming. R-1..R-3 will be re-measured with tuned thresholds.

### W-17: Baseline strategies retrain on 2048-row windows containing ~3-4 positives   (2026-08-27)
- **Lens:** research-validity
- **Finding:** With a 0.17% positive rate, a 2048-row window has ~3.5 positives on average. Partial retraining on that data alone is starved for positive signal. Even with the 30% replay buffer per D-8, a lot of the "fine-tune" step is noise.
- **Severity × Effort:** major × medium
- **Fix plan:** Increase window size to at least ~8192 (~14 positives) OR keep 2048 windows but batch multiple consecutive windows into the retrain data. Choose the option that mirrors real production practice (windows are 2048 for detection, retrain uses the last k windows).
- **Status:** open — decide after re-running Phase 1 with tuned thresholds; if numbers are still marginal, expand to option (b).

### W-18: Harness used `historical_X` as both training data AND the drift stream source   (2026-08-27)
- **Lens:** correctness
- **Finding:** `make_baseline_stream` applied drift to the same data the model was trained on. Under covariate shift on training data, F1 degrades honestly (the model saw x pre-drift, now sees x*1.5). But it meant we weren't testing "unseen data drifts" — we were testing "known data drifts." A reviewer will ask.
- **Severity × Effort:** major × low
- **Fix plan:** In `phase1_run.py`, use `X_stream` (a held-out 25% slice) as the stream, not `X_pre`. Confirm by rerunning and comparing pre- vs post-fix numbers.
- **Status:** fixed — `run_baseline` now takes `stream_X_raw` explicitly. Verified by re-running Phase 1 with tuned threshold and the held-out slice.

### W-19: Strategy instances were reused across seeds; internal `_last_retrain` state leaked between runs   (2026-08-27)
- **Lens:** correctness / research-validity
- **Finding:** `phase1_run.py` originally built `strategies = [PSIFullRetrainStrategy(), ...]` once outside the seed loop. Because `EWCOnlyStrategy` (and `PSIFullRetrainStrategy`) is a dataclass with mutable `_last_retrain` state, seed 0 sets it and seeds 1..N inherit that state — so subsequent seeds' cooldowns don't reset. Fingerprint evidence: EWC-only mean_f1 = 0.7539159... EXACTLY, four digits identical for seeds 1-4, because the trigger never fired again after seed 0.
- **Severity × Effort:** major × low
- **Fix plan:** Instantiate a fresh strategy inside the seed loop. Add a smoke test that verifies two consecutive `run_baseline` calls on the same strategy DO reset behavior.
- **Status:** fixed — `phase1_run.py` now uses factory list + fresh instance per seed.

### W-20: Only 5 seeds and 1 scenario; below the ≥10-seed × multi-scenario standard the paper commits to   (2026-08-27)
- **Lens:** research-validity
- **Finding:** Current gate runs 5 seeds × 1 scenario (amount_multiplicative_abrupt). W-5 already noted 10 is safer; the paper's H2/H3 tables should carry ≥5 scenarios × ≥10 seeds each.
- **Severity × Effort:** major × medium (compute is cheap, wall-clock isn't)
- **Fix plan:** After Phase 1's gate lands, re-run with 10 seeds × all 5 default scenarios before moving to Phase 2.
- **Status:** open — Phase 1's initial R-1..R-3 will be recorded at 5×1; the paper table uses 10×5.

### W-24: RSO's "no-op forever" convergence in R-4 — need scenarios where SLA is actually contested   (2026-08-28)
- **Lens:** research-validity
- **Finding:** In R-4 the RSO learns to never retrain because 4/5 scenarios have F1 above SLA even without action, and the cost weights make any retrain reward-negative. This is honest but leaves H2 partially untested (does RSO deliver Pareto improvements over baselines when the SLA IS contested?).
- **Severity × Effort:** major × medium
- **Fix plan:** Add scenarios where post-drift F1 sits *just below* SLA (~0.55 target vs 0.65 SLA) so no-op is definitely suboptimal and PPO must learn *when* to retrain. Also lift the SLA to 0.75 in a variant so most scenarios contest it.
- **Status:** **fixed** in commit `788b1b98` (step-A). `benchmarks/synthetic_drift_gen/injector.py::build_contested_sla_scenarios` — 3 scenarios calibrated so each of {partial, full, no-op} is the correct action somewhere. Wired into `phase_a_run.py`.

### W-26: Sensitivity sweep exited mid-run with no traceback — investigate memory/handle leaks between cells   (2026-08-28)
- **Lens:** product-readiness / correctness
- **Finding:** First attempt at `benchmarks/phase2_sweep.py` (9 cells, 3 seeds, 1500 timesteps) exited with code 4 mid-second-cell, no Python traceback in the log. Likely accumulated CUDA memory (retained PPO models, cloned adapters, torch tensors across cells) or an OS-level termination.
- **Severity × Effort:** major × low
- **Fix plan:** Added `gc.collect()` + `torch.cuda.empty_cache()` between cells and per-cell JSON checkpointing so mid-run crashes leave usable partial results. Retry ran successfully at 2 seeds × 800 timesteps × 10 windows.
- **Status:** fixed — retry succeeded, R-4b recorded.

### W-27: PPO exhibits bang-bang policies across the 3×3 sweep — no cell learned a state-conditional mixed policy   (2026-08-28)
- **Lens:** research-validity
- **Finding:** Every cell of R-4b's grid learned either "always no-op" or "always partial." Zero cells learned mixed / state-conditional policies. Classic RL pathology on tiny discrete action spaces + short training + sparse reward signal.
- **Severity × Effort:** blocker × high (must be fixed before H2 is called "tested")
- **Fix plan:** Combine several fixes in Phase 5: (1) longer training (30k+ timesteps), (2) contested-SLA scenarios (W-24), (3) richer state that includes drift-duration features, (4) consider continuous action (fraction-of-layers to retrain) instead of Discrete(3), (5) augmented-Lagrangian dual updates (W-22).
- **Status:** **partially fixed**. Fixes (1) longer training, (2) contested-SLA, (3) richer 15-d state, (5) Aug-Lagrangian dual updates all landed in commit `788b1b98` (step-A). R-Gate-A-smoke shows partial+full both fired during 500-timestep training (no more bang-bang). Fix (4) continuous action still deferred. Full-scale Gate A statistical verdict pending the multi-hour Colab run (see task R2).

### W-25: RSO's mean_f1 is bit-exact across seeds (0.7539 to 16 digits) — PPO is deterministic given seeds but eval env is fully deterministic too   (2026-08-28)
- **Lens:** correctness (cosmetic)
- **Finding:** Because PPO chose no-op every window, the eval env just replays FraudNet on a deterministic drifted stream — same F1 every seed. Fine, but it means seed variance is trivially 0 for RSO in this run, and Wilcoxon p-values are noisy at n=5 with zero-variance signal.
- **Severity × Effort:** minor × low
- **Fix plan:** Once RSO learns a mixed policy (W-24 fix), seed variance will re-appear.
- **Status:** open — will resolve once RSO's policy is non-trivial.

### W-22: Fixed-λ reward shaping ≠ true augmented-Lagrangian (D-16)   (2026-08-27)
- **Lens:** research-validity
- **Finding:** Phase 2's Lagrangian is fixed-λ (=1.0) reward shaping, per D-16, not the dual-update scheme the reference proposes. Fine for measuring the *value* of a cost-aware retrain policy, but a reviewer will point out that fixed λ can't adapt to changing constraint tightness.
- **Severity × Effort:** major × medium
- **Fix plan:** Phase 5 — implement augmented-Lagrangian update: `λ ← max(0, λ + η · (rolling_SLA_violation_rate − target))`, log λ over training.
- **Status:** **fixed** in commit `788b1b98` (step-A). `cadence/rso/lagrangian.py::AugmentedLagrangianEnv` wraps the sandbox and does dual ascent between resets. Verified in R-Gate-A-w28 where λ climbed 1.0 → 87.2 across training.

### W-23: Env's `_do_partial_retrain` does NOT apply EWC penalty; only the EWC-only baseline does   (2026-08-27)
- **Lens:** correctness / research-validity
- **Finding:** In the sandbox env, `partial` action calls `adapter.partial_fit(layers_to_update=["layer1"], replay_X=…)` with `ewc_penalty=0`. So the "partial" action is really "layer1-only fine-tune with replay, no EWC" — not "layer1-only EWC-regularized fine-tune" per the reference's Part 3B. This makes RSO's partial slightly less faithful to Claim C's protocol.
- **Severity × Effort:** major × low
- **Fix plan:** Add EWC computation to the env before the first partial, cache Fisher on the pre-drift model, apply penalty during the retrain. Same D-8 defaults as EWC-only baseline.
- **Status:** **fixed** in commit `788b1b98` (step-A). `SandboxConfig.ewc_penalty` + `_do_partial_retrain` compute Fisher on a historical slice, cache theta*, and pass ewc_penalty to `adapter.partial_fit`. Confirmed by tests/unit/test_lagrangian.py.

### W-21: "Negative forgetting" — unrelated-segment F1 IMPROVES after retrains — is not what we expected   (2026-08-27)
- **Lens:** research-validity
- **Finding:** All three baselines record forgetting ≈ -0.01 to -0.09 (i.e., unrelated F1 got BETTER after retraining, not worse). Two possible causes: (a) our "unrelated" segment is a random 5% slice of the same training distribution, not a genuinely held-out different task — so any retrain that keeps the classifier reasonable will move the threshold usefully; (b) threshold retuning after partial_fit adapts to new probability distribution and happens to give better F1 on the unrelated segment as a side effect.
- **Severity × Effort:** major × medium
- **Fix plan:** Redefine "unrelated" as a *distinct sub-population* of Fraud transactions — e.g., "high-Amount transactions only" — that is deliberately NOT drifted, so a retrain that overfits to the drift will visibly hurt unrelated F1. This is the honest H3 test.
- **Status:** **fixed** (two-step): (a) `cadence/data/disjoint.py::make_disjoint_split` in commit `384e9c77` (step-D) redefines "unrelated" as a genuinely disjoint sub-population (high_amount / late_time / class_positive). Gate D on Fraud was a negative result under this definition. (b) Commit `235d4d9d` (step-F) added `cadence/data/mnist_splits.py` for a genuinely-multi-task benchmark; Gate F reruns H3 on Split-MNIST tasks {0,1} vs {2,3} and gets partial < full forgetting at p=0.03125.

### W-28: PPO checkpoint from R-4 is trained on the pre-Step-A 13-d obs; new env emits 15-d (2026-08-29)
- **Lens:** research-validity + product-readiness
- **Finding:** Step A widened the RSO observation vector 13 → 15 (added `sla_margin` + `attribution_concentration`). The saved `experiments/rso_ppo_phase2.zip` was trained under the old obs. In Gate C's closed-loop episode and everywhere else the RSO is invoked, `phase_c_run` and `phase_e_run` transparently fall back to a rule policy because the load fails the obs-dim shape check. That means the RSO's *actions* everywhere post-Step-A are rule-driven, not learned.
- **Severity × Effort:** major × medium (this is what the pending full Gate A run is for)
- **Fix plan:** Re-train PPO on the 15-d obs with the Aug-Lagrangian wrapper + EWC-in-sandbox + contested scenarios enabled (the full Gate A protocol: 10 seeds × 30k timesteps × 5+ scenarios). Ship a new `experiments/rso_ppo_step_a.zip` and update the dashboard fallback logic to prefer it.
- **Status:** **fixed (engineering close-out)** in commit `60d97a8a`. `experiments/rso_ppo_phase_a.zip` (156605 bytes) trained at 1 seed × 800 timesteps on 3 contested scenarios (9962 s wall). Verified to load into `phase_c_run` without obs-dim mismatch. Full paper-scale multi-seed sweep is a separate concern tracked under task R2.

### W-29: Text drift (Amazon Reviews / Yelp) not measured; Step F Part IV missing (2026-08-29)
- **Lens:** research-validity
- **Finding:** The plan's Step F lists text-drift experiments alongside the tree/MNIST parts I ran. Text drift needs the Amazon Reviews 2023 archive (~7 GB) plus a text classifier (probably a small BERT distillation or an n-gram TF-IDF + LR). Neither is in-tree. Without it, the paper's "generality" claim covers tabular + Split-MNIST but not text.
- **Severity × Effort:** minor × high
- **Fix plan:** Add `cadence.data.text` loader for Amazon Reviews 2023, a `cadence.adapters.text.TFIDFLRAdapter`, and a Phase F Part IV runner. Deferred as follow-up.
- **Status:** **partially fixed** in commit `761bc47f`. `cadence/adapters/text.py::TFIDFLogRegAdapter` and `cadence/data/text_yelp.py` land plus `benchmarks/phase_f_text.py`. Yelp early → late by year showed the late slice is *easier* than train val (0.964 vs 0.947) so the drift claim on Yelp is untested. **Still open**: use a domain-shift split (restaurants → non-restaurants) that produces measurable degradation — tracked under task R5.

### W-30: Streamlit dashboard uses an illustrative CDAG layout, not the artifact's raw NOTEARS weights (2026-08-30)
- **Lens:** product-readiness
- **Finding:** `experiments/*_summary.json` files don't ship the raw NOTEARS-weighted adjacency matrix; only aggregated per-strategy metrics. The dashboard therefore renders a placeholder CDAG topology to communicate the SHAPE of what CADENCE produces, not the exact edges from any specific run. A stakeholder demo works; a customer-grade demo doesn't.
- **Severity × Effort:** minor × low
- **Fix plan:** Extend the JSON artifacts written by `phase_b_run` / `phase_c_run` with a `cdag_weights` list of `[[src, dst, weight], ...]` triples per (scenario, seed), and read that in the dashboard when present.
- **Status:** **open** — being addressed in the final push (task P1).

### W-31: docker-compose stack has never been run (no compose-time smoke test) (2026-08-30)
- **Lens:** product-readiness
- **Finding:** `docker-compose.yml` + `docker/Dockerfile.dashboard` are committed but the images have never been built on this machine. The clean-clone quickstart claim in `docs/product.md` is code-verified but not runtime-verified.
- **Severity × Effort:** minor × low
- **Fix plan:** `docker-compose up --build dashboard mlflow` on a clean clone; record R-Gate-G-clean-clone with the browser screenshot.
- **Status:** open — pending a clean-clone reboot.

### W-38: PPO is trained per (scenario, seed), not once per seed — Stage 1 budget blown 4-5x   (2026-09-02)
- **Lens:** research-validity / performance-cost / correctness
- **Finding:** In `benchmarks/phase_a_run.py` the per-seed-policies training call
  sits INSIDE the `for scenario in eval_scenarios:` loop (~L343-350). Under
  `--per-seed-policies` this trains a fresh PPO for every (scenario, seed)
  rather than once per seed. Consequences:
    1. **Compute blowup.** Stage 1 seed 0's first scenario PPO took 6985 s
       (~2 h). 3 contested scenarios × 3 seeds = 9 PPO trainings × ~2 h ≈
       **18 h** for Stage 1 — 4.5x the pre-registration's 4 h budget.
       Stage 2 (8 scenarios × 10 seeds × ~2 h) ≈ **160 h** locally.
    2. **Overfits policies to a single scenario** (no cross-scenario
       generalization signal). The multi-scenario env factory was designed
       to train ONE policy round-robin across scenarios, then evaluate on
       each — the intended §2.9-fixed curriculum.
    3. **Pre-registration says** "each independent training seed produces
       one policy; each policy runs deterministic eval over the 8 test
       scenarios" (docs/gate_a_preregistration.md Part 2, "Unit of
       analysis"). Current code violates that by producing 8 policies per
       seed, not 1.
- **Severity x Effort:** blocker x low. One-line fix: move the
  `if args.per_seed_policies: model = train_ppo(...)` block OUT of the
  `for scenario` loop and INTO an outer `for seed in seeds_iter` block
  that trains once per seed, then re-uses that model across all scenarios
  in the eval loops that follow.
- **Fix plan:**
    1. Refactor `benchmarks/phase_a_run.py` to hoist PPO training out of the
       scenario loop.
    2. Add a failing unit test that mocks `train_ppo` and asserts it is
       called exactly `len(seeds_iter)` times regardless of the number of
       scenarios.
    3. Rerun Stage 1 with the fix (should now complete in ~4-6 h).
- **Status:** **FIXED** (2026-09-02). Chose option (c): killed the in-flight
  buggy Stage 1 (5.5 h in, 0 seeds complete, λ pinned at the 100 cap → heading
  for a G6 failure anyway), backed the ledger up to
  `experiments/gate_a_ledger.pre-w38.json`, and hoisted the `train_ppo(...)`
  call out of the `for scenario in eval_scenarios` loop into a per-seed loop
  that runs ONCE before scenario evaluation (`benchmarks/phase_a_run.py`:
  training now at the `for seed in seeds_iter` block ~L296, reused via
  `models_by_seed[seed]` in the eval loop). Guard test
  `tests/unit/test_w38_per_seed_training.py` asserts `train_ppo` is called
  exactly `n_seeds` times regardless of scenario count (2 tests green).
  Because `_wrapped_factory` trains one policy round-robin across all contested
  scenarios, the per-scenario violation is now averaged rather than pinned on
  the unrecoverable scenario — this should also relieve the λ→100 saturation
  seen in the buggy run; watch gate G6 on the clean relaunch to confirm.

### W-39: dual_lr=5.0 saturates λ at cap under W-32 rescaled reward (fixed)   (2026-09-03)
- **Lens:** research-validity / correctness
- **Finding:** Stage-1 diagnostic run (`gate-a-cbb3f72d`) failed pre-reg gates G6, G7, G8. Root cause: `AugmentedLagrangianConfig.dual_lr=5.0` + `target_violation=0.0` under the W-32-rescaled reward (cost_scale=1e7) pushed lambda_sla to the lambda_max=100.0 cap early in training and held it there for all 9,216 dual_update events in the last 30% of training across all 3 seeds. Under saturated λ, the policy collapses to always-retrain (deterministic → G7 fail; expensive → G8 fail).
- **Severity × Effort:** blocker × trivial (one default per file).
- **Fix plan:** Calm both defaults to `dual_lr=1.0`:
    1. `cadence/rso/lagrangian.py::AugmentedLagrangianConfig.dual_lr`
    2. `benchmarks/phase_a_run.py --dual-lr`
  Add `tests/unit/test_w39_dual_lr_calmed.py` pinning both defaults + the unchanged other dual-update knobs.
- **Status:** fixed. See commits landing this weakness + R-Gate-A-stage1-fail-1. Stage-1 re-run pending user authorization (~5h wall).
