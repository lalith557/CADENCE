# Gate A — pre-fix audit (§3 of the master research prompt)

**Date:** 2026-08-31. **Author:** current agent, wearing all four hats
(RL researcher / ML-systems engineer / experimental scientist / hostile
paper reviewer). **Rule:** no code changes yet — this document is the
diagnosis. Every §2 lead is verified or falsified below with a citation
to the actual code and, where feasible, an empirical measurement.

The bottom line up front: **five load-bearing bugs** are confirmed and
one is the almost-certain root cause of the "policy never learns to
save cost" collapse.

---

## 1. Confirmed facts (verified via code read + empirical checks)

### 1.1 Reward scaling is broken by 9 orders of magnitude (load-bearing bug)

**Code:** `cadence/rso/env.py:210-217`.
```
reward = (delta_f1
          - self.cfg.w_gpu_hr * cost_gpu_hr
          - self.cfg.w_kg_co2 * cost_kg_co2
          - self.cfg.lambda_sla * sla_penalty)
```
FraudNet has `flops_per_forward ≈ 4100`. Empirically measured on
`configs/default.yaml` (`w_gpu_hr=0.05, w_kg_co2=0.5, lambda_sla=1.0`):

| term | typical magnitude |
|---|---:|
| `Δf1` (a real retrain recovering 5 F1 points) | 5e-2 |
| `λ · sla_penalty` (10 F1 points below SLA) | 1e-1 |
| `w_gpu_hr · gpu_hr` (partial, `finetune_epochs=1`) | **7.5e-11** |
| `w_gpu_hr · gpu_hr` (full, 15 epochs) | **1.2e-8** |
| `|sla_penalty| / |cost_penalty_partial|` ratio | **1.3 × 10⁹** |

The compute-cost term is **effectively zero in the reward**. PPO cannot
possibly learn a cost-aware policy because the gradient signal from
cost is 10⁹× smaller than the noise floor of any other term. This is
the almost-certain root cause of the "bang-bang" collapse in R-4, R-4b,
R-Gate-A-smoke, and every downstream Gate-A attempt. Fix must land
before any confirmatory run.

### 1.2 PPO `ent_coef` unset → default 0.0

**Code:** `cadence/rso/ppo.py:41-53` (dataclass), `95-107` (PPO
constructor). `PPO(cfg.policy, vec_env, n_steps=..., batch_size=..., ..., seed=...)`
does not pass `ent_coef`, so SB3 uses its default of 0.0. Zero-entropy
PPO on a tiny `Discrete(3)` action space with short training is the
canonical recipe for premature deterministic collapse. R-4b's finding
that every cell went bang-bang is consistent with this. Fix: set
`ent_coef` (e.g. 0.01) and add it to `PPOTrainConfig`.

### 1.3 No observation or reward normalization

**Code:** `cadence/rso/ppo.py` — the `DummyVecEnv` at line 94 is not
wrapped in `VecNormalize`. The observation vector spans:

| index | field | range |
|---|---|---|
| 0-4 | top-5 scores | [0, 1] |
| 5,6 | F1, SLA | [0, 1] |
| 7 | windows_since_alert | [0, 20] |
| 8-10 | per-action gpu_hr | [0, 100] |
| 11 | grid intensity g/kWh | [0, 2000] |
| 12 | hours_since_retrain | [0, 10000] |
| 13, 14 | sla_margin, concentration | [0, 1] |

That's a **~10⁷ dynamic range** on the raw MLP input. An MLP without
BatchNorm/LayerNorm and no observation normalization will treat the
grid intensity and hours-since-retrain as dominating features by
sheer magnitude. `VecNormalize` (obs + reward) is the fix.

### 1.4 Training reward is in-sample F1; eval reports prequential F1

**Code path:** `env.step` (`env.py:157-234`) evaluates `post_f1` on the
**same window** the retrain was fit on (in-sample); `evaluate_ppo`
(`ppo.py:182-193`) appends `info["pre_f1"]` to `rolling_f1` and reports
its mean (prequential — F1 before the retrain's effect). Confirmed by
reading both functions. This means **the training reward is optimistic
by exactly the amount a small model overfits a 2048-row window**, and
PPO is optimizing a different, inflated objective than the one every
R-entry actually reports. Prequential is the honest choice; make
training and eval consistent.

### 1.5 "10 seeds" reuses a single seed-42 policy — pseudoreplication

**Code:** `benchmarks/phase_a_run.py:223-232` trains PPO once
(`PPOTrainConfig(..., seed=42)`) *outside* the seed loop at line
286-338, which only varies `make_baseline_stream(..., seed=seed)` for
the eval stream and `set_global_seed(seed)`. The 10 "seeds" in every
Gate-A run share one policy checkpoint. **W-25's bit-exact F1
across seeds** is the direct evidence: PPO acted identically on each
seed's eval stream because the policy was identical.

Consequence: reported n=10 samples are **not independent** for the
"RSO training variance" claim. The paper's unit of analysis must
either be (a) independent-policy seed (train once per seed) or
(b) explicit hierarchical design that models the two nested factors.

### 1.6 Replay buffer uses `np.random.default_rng(0)` every call

**Code:** `env.py:254` — inside `_do_partial_retrain`, a fresh
`np.random.default_rng(0)` is instantiated per invocation. Every
partial retrain, in every step, in every episode, in every run, sees
the **exact same replay batch** from `historical_X`. This turns the
replay buffer into a deterministic constant and contributes to the
zero-variance issue.

### 1.7 Train/eval fidelity mismatch

**Code:** training `SandboxConfig` in `phase_a_run.py:174-190` uses
`finetune_epochs=1, fullretrain_epochs=1` ("fast sandbox"); eval
`SandboxConfig` at `phase_a_run.py:260-275` uses `5, 15`. Cost
`estimate_cost(..., epochs=1)` and `estimate_cost(..., epochs=15)`
differ by 15×; recovery-F1 dynamics differ by roughly the same factor.
PPO is optimising against a cost/recovery structure it will never
face at eval time. Justify or remove the mismatch — either honest
per-scenario simulation at training time, or a documented "surrogate"
training reward that reviewers can defend.

### 1.8 Aggressive dual-λ

**Code:** `lagrangian.py:39-48`. Defaults `dual_lr=5.0, lambda_max=100.0,
target_violation=0.0`. Prior R-Gate-A-w28 log showed λ climbed to 87.2
by the end of training — near the cap. `target_violation=0` is strict:
any SLA gap grows λ. Combined with a reward where cost is already
invisible (see 1.1), λ inflating pushes the policy to "always
retrain," throwing away the entire cost objective.

Fix candidates: (a) set `target_violation` to something >0 (e.g., 0.02
allowed slack), (b) drop `dual_lr` to 0.5–1.0, (c) fix reward scaling
first — under a scaled reward, dual-λ may not need to be so aggressive.

### 1.9 "Random scenario per reset" is a fixed round-robin (documentation bug)

**Code:** `ppo.py:263-287`. Comment says "random scenario per reset,"
but the factory uses `counter["i"] % len(cached)` and `rng` is unused
(`_ = rng  # keep for future randomization`). Training sees scenarios
in a fixed rotation, not randomised. Not strictly a bug (rotation
covers scenarios evenly), but the docstring lies and the "curriculum"
claim in the paper skeleton is wrong.

### 1.10 Single `DummyVecEnv` — no parallelism

**Code:** `ppo.py:94`. A single-thread `DummyVecEnv` is the slowest
possible vectorization. `SubprocVecEnv(num_envs=k)` with `k=4-8` gives
approximately linear PPO wall-time speedup with no science change.
Legitimate §5 efficiency win.

---

## 2. Likely bugs (unconfirmed, medium confidence — flagged for testing)

### 2.1 Fisher cache invalidation on reset

`env.py:146-148` — reset() explicitly does NOT invalidate Fisher, on
the argument "adapter was just reset to the same initial_state." This
is correct IF the adapter is reset before any partial retrain. But if
a partial retrain succeeded and the adapter's weights changed, and
then reset() is called, the cached Fisher is stale relative to the new
`_initial_state` snapshot (which is unchanged). Actually reading more
carefully: `_initial_state` never changes — it's captured in `__init__`.
So Fisher stays valid. But this deserves an explicit test.

### 2.2 Terminated vs truncated semantics

`env.py:219-220`: `terminated = post_f1 >= sla_target`; `truncated =
windows_since_alert >= max_windows`. Under an SLA-recovery termination,
episodes end early — training reward truncates. If SLA is easily met
(current default 0.65 with a baseline F1 of ~0.83), most episodes end
after the first step and PPO learns essentially nothing about the
downstream cost/recovery trajectory. This interacts with the reward
scaling bug: `Δf1` on step 0 is dominated by *the pretrained model's*
undrifted-window F1, not by any retrain the agent chose.

### 2.3 `evaluate_ppo` records `pre_f1` — off-by-one

Well-intentioned "match the baseline harness protocol" but confusing:
if the agent chose `partial` at window t, that action's benefit
appears at window t+1 in the log (as `pre_f1[t+1]`), not at t. Not a
bug per se — just a metric definition worth documenting.

---

## 3. Likely weaknesses (design smells, not bugs)

- **`train_ppo` accepts a raw `env_factory` (`ppo.py:78-124`) and wraps
  it in `DummyVecEnv`.** Callers who want `VecNormalize` have no hook.
- **`RSOEvalOutcome` has no `constraint_violation` field.** SLA
  violations count per-window but the aggregated field name
  `sla_windows_below` conflates "how many windows below SLA" with
  "average magnitude of violation."
- **No held-out validation set for HP tuning.** Gate A currently
  trains + evals on the same scenarios. Any HP choice is implicitly
  tuned on the test scenarios.
- **The `contested_*` scenarios were tuned by inspection** (see
  R-Gate-A-smoke). Reviewer will ask "did you tune the scenarios so
  RSO wins?" Answer needs a scenario-generation protocol that runs
  before RSO training.

---

## 4. Missing experiments (things a reviewer will demand)

- **Reward-component sweep**: with rescaled cost, what
  (w_gpu_hr, w_kg_co2, lambda_init) gives learnable behaviour?
- **Ablations §14**: without contested scenarios, without Aug-Lagrangian,
  without ent_coef, without VecNormalize, without the 15-d enrichment.
- **Robustness §15**: drift magnitude sweep; compute-price sweep;
  SLA-target sweep.
- **Real independent seeds §13**: retrain the policy per seed; measure
  the variance we've been pretending we had.

---

## 5. Reproducibility risks

- **No fixed seed for `env_factory` closures.** `_wrapped_factory` in
  `phase_a_run.py:219-221` closes over `base_factory` which closes
  over `seed=0`. If any caller changes seed=0 → different envs.
- **MLflow file store used to be the default.** Now sqlite (P5 fix).
  Older `experiments/mlruns/` directories may not round-trip.
- **Windows-only path separators** in some Colab driver comments.

---

## 6. Evaluation risks

- **In-sample training reward vs prequential eval metric** (already in §1.4).
- **`terminated = f1 >= sla_target` in eval** — if `run_full_episode=False`,
  eval stops the moment the policy trivially met SLA on the pretrained
  model. Fixed by `run_full_episode=True` in `evaluate_ppo(..., run_full_episode=True)`
  which is the default, but worth guarding.
- **Baseline strategies use their own harness** (`benchmarks/baselines/harness.py`),
  not the sandbox env. Verify they see the same drifted stream at the
  same window boundaries.

---

## 7. Training risks

- **Zero entropy + tiny action space + short training** = deterministic
  collapse (§1.2).
- **No advantage normalization** (SB3 PPO's default `normalize_advantage=True`
  is on, but check).
- **No value-function clip** either; SB3 default handles this.

---

## 8. Improvement opportunities

Ordered by (expected impact) / (implementation effort):

1. **Rescale reward so cost is comparable to Δf1.** Multiply the cost
   terms by `1e6 – 1e8` (an operator-chosen scaler that puts the
   full-retrain cost around 0.1–0.3 F1-points). This alone likely
   restores a learnable cost signal. **Highest impact, lowest effort.**
2. **Set `ent_coef=0.01`** in `PPOTrainConfig`. Prevents deterministic
   collapse.
3. **Wrap the training env in `VecNormalize`** (obs=True, ret=True,
   clip_obs=10). Normalises observations and returns.
4. **Prequential in-training reward** — compute `post_f1` on the *next*
   window, not the current one. Matches eval metric; kills the in-sample
   bias.
5. **Independent per-seed policies** — train PPO inside the seed loop
   with `seed=seed_index`, not once outside with `seed=42`.
6. **Replay RNG per env instance** — `self._replay_rng = np.random.default_rng(...)`
   in `__init__`, use it in `_do_partial_retrain`.
7. **`SubprocVecEnv(4)` if OS allows** (Windows spawn is finicky —
   fallback to `DummyVecEnv` on failure). Cheap PPO speedup.
8. **Loosen dual-λ**: `dual_lr=0.5, target_violation=0.02`. Only after
   reward is rescaled.
9. **Held-out val scenarios for HP tuning.** Generate a *third* set of
   scenarios never used for training or final eval.

---

## Fix roadmap (with explicit test-first order)

For each of the 6 confirmed bugs in §1, the plan is:

1. Write a failing unit test that exercises the bug (`tests/unit/test_rso_audit.py`).
2. Apply the smallest fix.
3. Show the test passes.
4. Record as W-32..W-37 in `docs/weaknesses.md`.
5. Commit each independently so bisection stays trivial.

Priority order (highest impact first):
1. W-32: reward scaling (§1.1) — the load-bearing bug.
2. W-33: `ent_coef` (§1.2).
3. W-34: `VecNormalize` (§1.3).
4. W-35: prequential training reward (§1.4).
5. W-36: per-seed policies (§1.5).
6. W-37: replay RNG (§1.6).

Then Stage 1 diagnostics (§23) at reduced scale, then pre-register success
(§24), then confirmatory Stage 2 at full scale — the last of which likely
needs Colab.
