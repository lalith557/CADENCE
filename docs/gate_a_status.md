# Gate A — Rolling Status (progress across sessions)

> Updated by whichever session runs `run_experiment.py` or lands work below.
> This is the single-source-of-truth "where are we." Ledger lives at
> `experiments/gate_a_ledger.json`; per-seed results under `experiments/gate_a_stage{1,2}/`.

**Last touched:** 2026-09-01 (this session — see §"This session" below).

---

## Protocol sections — DONE / PARTIAL / NOT-DONE

Verified against actual code / results / artifacts, not commit messages.

| § | Task | Status | Evidence |
|---|---|---|---|
| §0 | Honesty contract acknowledged | DONE | pre-reg part 3 anti-cheating clauses |
| §1 | Role + prime directives | DONE | Protocol frozen in-repo |
| §2 | Verify §2 audit leads with failing tests | DONE | `tests/unit/test_rso_audit.py` (7 tests green) |
| §3 | Audit deliverable | DONE | `docs/gate_a_audit.md` (5 confirmed load-bearing bugs + roadmap) |
| §4 | Resumable overnight pipeline | **DONE (this session)** | `run_experiment.py` + atomic ledger + `tests/unit/test_run_experiment.py` (3 green) |
| §5 | Legitimate efficiency wins | PARTIAL | `SubprocVecEnv` not attempted (§1.10 audit); OK for Stage 1 |
| §6 | Repair PPO training (W-32..W-37 fixes) | DONE | commits `d6e7121c`, `20b8ecac`, `229b629e`, `52be880e`, `2d382be9`, `d553c497` |
| §6 | Reward-decomposition instrumentation for gates G3/G4/G5 | **DONE (this session)** | commit `d3a3406`, `env.step` info dict |
| §7 | Contested-SLA scenarios | DONE | `build_contested_sla_scenarios` in `benchmarks/synthetic_drift_gen` |
| §8 | Data-leakage / TRAIN-VAL-TEST | PARTIAL | pre-reg §5 documents split; runner enforces contested-only for Stage 1 |
| §9 | Fair, un-weakened baseline suite | DONE | PSI+full / fixed / EWC-only in `benchmarks/baselines` |
| §10 | Complete Gate A (10 seeds × 30k × frozen protocol) | **STAGE 2 NOT RUN** | runner ready; awaits Stage 1 gate pass |
| §11 | Pareto / trade-off surface | PARTIAL | `phase_a_run` emits Wilcoxon table; Pareto plot pending post-Stage-2 |
| §12 | Gate E scientific reconsideration | PARTIAL | pre-reg §12 lists three claim levels; §6 below is the fallback headline |
| §13 | Statistical rigor / unit of analysis | DONE | pre-reg §3; `--per-seed-policies` enforces independent policies |
| §14 | Ablations | PARTIAL | components exist; unified ablation table pending |
| §15 | Robustness | PARTIAL | R-Gate-F robustness curves landed; drift-magnitude sweep pending |
| §16 | Elec2 re-verification from raw | **DONE (prior session)** | `R-Gate-E-verified` in `docs/results.md` — 69.14%/-0.017 REPRODUCES; classified PRACTICAL (Claim B) |
| §17 | Hidden-bug hunt | DONE | audit §1 covers 10 confirmed/suspected bugs; W-32..W-37 fixed |
| §18 | Automated sanity checks | PARTIAL | 7 audit tests + 3 runner tests; §17 broader guards pending |
| §19 | One-command experiment | **DONE (this session)** | `run_experiment.py` with `--stage`, `--seed`, `--resume`, `--dry-run` |
| §20 | Publication-quality outputs (plots/tables) | PARTIAL | auto paper-skeleton exists; Pareto/robustness plots pending |
| §21 | Experiment ledger | **DONE (this session)** | `experiments/gate_a_ledger.json` with proper status enum |
| §22 | Reproducibility manifest | PARTIAL | git+config hash in ledger; environment lockfile pending |
| §23 | Two-stage strategy (Stage 1 gate) | **STAGE 1 NOT RUN** | runner ready; launch this session |
| §24 | Pre-register success | DONE | `docs/gate_a_preregistration.md` signed 2026-09-01, commit `2d382be9` (pre-rewrite `cfcf860`) |
| §25 | Final deliverable (A–L) | PENDING Stage 2 | writes on Gate A completion |
| §26 | Guardrails recap | ACTIVE | pre-reg Part 3 |

## Pre-registration Stage-1 / Stage-2 runs

| Stage | Protocol | Status | Ledger |
|---|---|---|---|
| Stage 1 (diagnostic) | 3 seeds × 3000 steps × 3 contested scenarios; 8 G-gates | **launched this session (background)** | `experiments/gate_a_ledger.json` |
| Stage 2 (confirmatory) | 10 seeds × 30000 steps × 8 scenarios; frozen | **NOT_STARTED** — blocked by Stage 1 pass | ditto |

## Product-launch gaps

| Task | Status | Evidence / blocker |
|---|---|---|
| Docker boot (`docker-compose up --build`) | NOT DONE | Docker daemon offline last check (see R-Gate-G-clean-clone); needs Docker Desktop running |
| Real dashboard CDAG (W-30) | NOT DONE | requires extending artifacts w/ raw NOTEARS weights + rendering |
| REST API end-to-end verification | **DONE (this session, partial)** | 6/7 `tests/integration/test_api.py` pass under FastAPI TestClient (`/score`, `/monitor`, `/decide` rule branch, `/health`, `/admin/reload`); 1 SKIPPED — `test_api_learned_ppo` fails to load the existing PPO checkpoint with `No module named 'numpy._core.numeric'` (numpy-checkpoint compatibility, not an API bug). Learned branch will re-verify once the Stage 1 PPO checkpoint under the current numpy is available. |
| CI green on real push (W-8) | NOT DONE | requires `git push`; no green badge yet |
| Remove `mypy continue-on-error` (W-11) | NOT DONE | after typing pass |

---

## This session (2026-09-01)

Delivered (bounded engineering finished):
- `run_experiment.py` — one-command resumable runner (§19).
- `experiments/gate_a_ledger.json` schema — atomic writes, per-seed status transitions
  `NOT_STARTED → RUNNING → COMPLETED / FAILED / INTERRUPTED` (§21).
- `--only-seed N` added to `benchmarks/phase_a_run.py` for subprocess-per-seed drive.
- `tests/unit/test_run_experiment.py` (3 tests, all green): atomic ledger writes,
  resume skips COMPLETED + picks up INTERRUPTED, SIGINT marks INTERRUPTED (never
  PASS/FAIL, per pre-reg Part 3).
- Reward-component decomposition surfaced in `env.step`'s `info` dict — required by
  Stage-1 gates G3/G4/G5. Test in `test_rso_audit.py` pins the contract.
- `configs/gate_a.yaml` committed (deferred to the signed pre-reg on any conflict).

Launched / in-flight: **Stage 1** — see §Live below.

**Resume command (for the next session or after a shell restart):**
```
python run_experiment.py --config configs/gate_a.yaml --stage 1 --resume
```

## 2026-09-03 session — 4 tasks landed (Stage 1 relaunched in-flight)

Executing the user's ordered task list:
1. **G1-G5 callback wired (W-40)** — commit `86a421e`. `RewardComponentCallback`
   aggregates env.step's info dict + action distribution and logs to MLflow
   with metric names `reward/*` and `action/*`. `train_ppo` now uses
   `CallbackList([MLflowCallback, RewardComponentCallback])`. 2 tests
   (`tests/unit/test_w40_reward_component_callback.py`) — metric names +
   missing-key robustness — both pass; 17/17 total suite green.
2. **Stage 1 relaunched** with W-38 (per-seed hoisted) + W-39 (dual_lr=1.0) +
   W-40 (reward callback) — background task `b3gwmak7r`, log at
   `logs/stage1_w39w40.log`. Ledger `experiment_id: gate-a-86a421e-…`.
3. **Evaluator extended** to actually query MLflow for G1-G5 — commit
   `256f52f`. `scripts/evaluate_stage1_gates.py` now grows real query paths
   for each gate; falls back to TBD-with-reason on missing metrics; filters
   stale runs by `ledger['created']` start_time. Preliminary against the
   in-flight run: **G6 PASS at final λ=3.58 (36 dual events, max ever 3.58 —
   dramatic improvement over W-39-precursor's 100 saturated).** G1-G5 correctly
   TBD until PPO training starts logging reward-component metrics.
4. **Colab handoff recipe** (`scripts/colab_gate_a.py`, commit `39cc4b6`) —
   replaced the pre-W-38 Aug-30 driver with a 7-cell paste-into-Colab
   notebook recipe that wraps the current resumable runner unchanged.
   Anti-cheat section mirrors pre-reg Part 3 rules 1-6.

Once Stage 1 finishes and all 8 gates evaluate: if PASS, hand off Stage 2
via that Colab recipe (Stage 2 at ~20h wall is not laptop-friendly).

## Prior session — bounded work landed (2026-09-02, post W-38 fix)

- **W-38 (previous session's log)**: fixed by commit `cbb3f72d` — `train_ppo` hoisted
  out of the scenario loop; `models_by_seed[seed]` reused across all scenarios;
  new `tests/unit/test_w38_per_seed_training.py` (2 tests, both pass).
- **API test robustness (numpy checkpoint compat, W-11 tail)** — commit `d3a3406→(current)`:
  `test_decide_uses_learned_policy_when_checkpoint_loaded` now tries a *list* of
  candidate checkpoints (per-seed from Stage 1 first, then legacy shared) and
  skips only if all fail. Removes the 7th-test skip once Stage 1 produces fresh
  per-seed checkpoints under the current numpy 1.26.
- **`.gitignore` hygiene** — mlflow.db + backups + WAL, gate_a_ledger*.json,
  gate_a_stage{1,2}/, logs/, `*.log`, gate-eval reports. `experiments/mlflow.db`
  untracked from index (kept locally).
- **`scripts/evaluate_stage1_gates.py`** — 8-gate reader per pre-reg Part 1.
  Auto-evaluates G6 (dual-λ trajectory from log), G7 (cross-seed metrics
  variance), G8 (wall time budget). G1-G5 marked TBD with exact MLflow
  queries named. Preliminary run against the in-flight Stage 1 log:
  **G6 PASS** — 46 dual_update events, final λ=19.07, max ever=19.07 —
  well below the 50 threshold and dramatically better than the pre-fix run
  that hit 87 near the 100 cap. Direct evidence W-32 (reward rescale) is
  working.
- **W-4 correction**: `docs/prior_art.md` is actually 197 lines of
  paper-ready related work with per-claim narrowing (D-17/D-18). Two ⚠️
  deep-reads deferred but non-blocking after narrowing. Not "NOT DONE" —
  substantially DONE, only the WebFetch-blocked deep-reads remain.

## Live (updated by runner)

**Stage 1 launched 2026-09-02 08:39 UTC. First launch failed at MLflow alembic
migration** (stale `experiments/mlflow.db` from Aug 31 held a revision the
installed MLflow 2.19 couldn't locate). Root cause: an mlflow db from a newer
version was left in place. **Fix landed same session:** the stale db was moved
to `experiments/mlflow.db.bak-YYYYMMDD-HHMMSS` (never deleted) and Stage 1
re-launched with `--resume`. The runner correctly marked seed 0 FAILED, stopped,
and the resume path picked it up.

**Re-launch as of 14:11 UTC:**
- Ledger transitions: seed 0 FAILED → seed 0 RUNNING → (seeds 1, 2 NOT_STARTED).
- Seed 0's PPO training is live on cuda (device_info logged), reward-component
  instrumentation active, augmented-Lagrangian dual-λ evolving healthily
  (3.2 → 9.6 over the first several dual updates — well below the prior run's
  runaway to 87). Both `action_partial` and `action_full` fire — no action
  collapse visible so far.
- Wall clock projection: ~30–60 min/seed on RTX 4070 Laptop; Stage 1 total
  ~1.5–3 hours end-to-end, within the pre-reg's 4-hour Stage 1 budget.

**Everything is checkpointed. The next session (or a fresh shell) resumes with:**
```
python run_experiment.py --config configs/gate_a.yaml --stage 1 --resume
```

When Stage 1 completes:
1. Read the per-seed results in `experiments/gate_a_stage1/seed_{0,1,2}.json`.
2. Evaluate the 8 G-gates from `docs/gate_a_preregistration.md` Part 1 against
   the reward-component decomposition (in each seed's MLflow run + step info).
3. Write outcome as `R-Gate-A-stage1` in `docs/results.md`. If all 8 pass,
   flip to Stage 2 via `python run_experiment.py --stage 2 --resume` (10 seeds
   × 30k steps × 8 scenarios; expect multi-hour to overnight; consider
   `scripts/colab_gate_a.py` if local budget is tight).
4. If any gate fails: single-fix loop per pre-reg Part 1's decision tree.

**§16 Elec2 re-verify: DONE this session** → `R-Gate-E-verified` in
`docs/results.md`. Recomputation from raw independently reproduces the
69.14% compute savings / -0.0166 F1 gap. Under pre-reg Part 2 mapping this is
a **PRACTICAL (Claim B)** result — cannot be upgraded to STRONG (Claim A)
without paired non-inferiority stats at n≥10.

**§7 Docker boot: BLOCKED** — Docker daemon is offline
(`failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine`).
User needs to start Docker Desktop. Then:
```
docker-compose up --build dashboard mlflow
# browse http://localhost:8501 for the Streamlit dashboard
```
Config was already validated (see R-Gate-G-clean-clone); this session did not
change the compose spec.
