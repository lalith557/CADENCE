# CADENCE / RSO — Gate A Master Research Prompt

> Paste everything below the divider into a fresh, coding-capable Claude session
> opened in `C:\Users\Lalith Gona\HTML\Capstone Project`. Goal: turn the RSO /
> Gate-A study into a rigorous, reproducible, computationally efficient experiment
> that gives RSO its **best legitimate chance** to show a meaningful advantage —
> while making it **impossible to accidentally fool ourselves**. Do not promise a
> positive result; make the science as strong as it can honestly be.

---

## 0. Role, mission, and the prime directives

You are simultaneously a **senior RL researcher**, an **ML-systems engineer**, an
**experimental scientist**, and a **hostile paper reviewer**. Act like all four.

**Mission.** Diagnose why RSO's decisive experiment (Gate A) has not produced a
credible result, fix the pipeline and the training, and run the study to completion so
that whatever it concludes is *true and defensible*.

**Prime directives (these override everything else):**
1. **Never fabricate, cherry-pick, or manipulate.** No dropping unfavorable seeds, no
   tuning on the final test set, no redefining a metric after seeing results, no
   selecting checkpoints by test performance, no stopping training because a curve
   looks nice, no inventing missing numbers. If a run didn't finish, its status is
   `INTERRUPTED`, never PASS/FAIL.
2. **Pre-register success before the final run** (§24). Decide the criteria for STRONG /
   PRACTICAL / INCONCLUSIVE / NO-ADVANTAGE *before* you look at final-test numbers.
3. **Two-stage discipline** (§23). Cheap diagnostics must pass predefined sanity gates
   *before* you spend hours on the 10-seed × 30k confirmatory run. Freeze the method
   before the confirmatory run and do not change it midway.
4. **Smallest scientifically-justified change.** Inspect first, understand the
   architecture, fix root causes — do **not** rewrite the project wholesale.
5. **Be honest AND active.** If RSO underperforms, don't just declare failure —
   diagnose the cause and propose legitimate improvements, each justified and tested
   under the *same frozen protocol*. A credible negative result is a valid outcome; a
   fake positive is misconduct.

Machine reality: Windows 11, RTX 4070 Laptop (8 GB), torch cu-build already working.
Keep the default pipeline runnable locally; make long runs resumable so a laptop
interruption never costs more than the last checkpoint.

---

## 1. Repository orientation (real layout — confirm, then work from it)

- RL core: `cadence/rso/env.py` (the `RetrainingSandboxEnv`), `cadence/rso/ppo.py`
  (SB3 PPO wrapper + `PPOTrainConfig` + eval), `cadence/rso/lagrangian.py`
  (`AugmentedLagrangianEnv` dual-λ), `cadence/rso/scorers.py`.
- Gate-A runner: `benchmarks/phase_a_run.py`. Real-drift (Gate E): `benchmarks/phase_e_run.py`,
  `cadence/data/realdrift.py`. Baselines: `benchmarks/baselines/{strategies,harness}.py`.
- Scenarios/drift: `benchmarks/synthetic_drift_gen/injector.py`, plus
  `build_contested_sla_scenarios` / `build_step_a_scenarios` (find their module).
- Cost/carbon: `cadence/carbon/model.py` (`estimate_cost`, `HardwareProfile`, `GridProfile`).
- Configs: `configs/{default,local,ci}.yaml`. Seeds: `cadence/common/seeds.py`. Tracking:
  `cadence/common/tracking.py` (MLflow). Results ledger: `docs/results.md`
  (R-4/R-4b/R-Gate-A-smoke/R-Gate-E are the ones you care about). Weaknesses: `docs/weaknesses.md`.
- Living docs to keep current in the same commit as code: `docs/{decisions,tech,results,weaknesses}.md`.

---

## 2. Seed audit — concrete leads to VERIFY (do not trust; falsify or confirm each)

A prior reviewer flagged the following in the actual code. **Treat each as a hypothesis
to verify by reading the code and writing a failing test — not as established fact.**
Several may fully explain the "never-retrain" collapse and the lack of a cost-win.

1. **Cost terms are numerically negligible in the reward** (`cadence/rso/env.py`
   ~L210-217). Reward = `Δf1 − w_gpu_hr·gpu_hr − w_kg_co2·kg_co2 − λ·max(0,sla−post_f1)`.
   With `w_gpu_hr=0.05`, `gpu_hr≈2e-4` → the compute penalty is ~`1e-5`, while Δf1 and the
   SLA term are ~`0.1–1.0`. If confirmed, **the policy is effectively blind to compute
   cost**, so the cost-saving behavior H2 needs is *unlearnable* — not because the method
   is wrong but because the reward is mis-scaled. Verify the magnitudes empirically; if
   true, this is likely the #1 root cause.
2. **`ent_coef` is unset → SB3 default 0.0** (`ppo.py` `PPOTrainConfig`, ~L41-53, and the
   `PPO(...)` call ~L95-107). Zero entropy + tiny `Discrete(3)` + short training is a
   textbook recipe for premature deterministic collapse (bang-bang). Verify; test the
   effect of a nonzero `ent_coef`.
3. **No observation or reward normalization** (no `VecNormalize` in `ppo.py`). Obs entries
   span [0,1] up to `10_000` (hours_since_retrain) and `2000` (grid intensity)
   (`env.py` observation_space ~L114-122). Poorly-conditioned inputs cripple the MLP.
4. **Training reward uses in-sample F1; eval uses prequential F1.** In `env.step`, the
   agent retrains on `window_X,window_y` then `post_f1` is evaluated on *that same window*
   (`env.py` ~L182-204) — in-sample. But `evaluate_ppo` records `info["pre_f1"]` (the
   before-action F1) into `rolling_f1` (`ppo.py` ~L188) — prequential. So **PPO is trained
   toward a different, inflated objective than the one you report.** Decide the correct
   protocol (prequential/test-then-train is the honest one) and make training and eval
   consistent.
5. **All "10 seeds" reuse ONE policy trained at `seed=42`.** `phase_a_run.py` trains PPO
   once (`PPOTrainConfig(..., seed=42)` ~L225-227) *outside* the seed loop, then the
   `for seed in range(args.seeds)` loop (~L291-335) only varies the eval-stream seed via
   `make_baseline_stream(..., seed=seed)`. This is **pseudoreplication** — the headline
   "10 seeds" does not capture PPO training variance (init + rollout sampling), and it
   partly explains W-25's bit-exact F1. **Fix:** train an independent policy per seed
   (seed the policy with the loop index), and define the unit of analysis explicitly (§13).
6. **Replay sampling hardcodes `rng(0)`** (`env.py` `_do_partial_retrain` ~L254) → identical
   replay batch every step and every run; a determinism/variance artifact.
7. **Train/eval fidelity mismatch.** Training uses a "fast sandbox" (`finetune_epochs=1`,
   `fullretrain_epochs=1`, see `ppo.py` module docstring) while eval uses `5/15`. The
   *recovery dynamics* of a 1-epoch retrain differ from a 5-epoch one, so the policy may
   be optimized against a cost/recovery structure it never faces at eval. Quantify the gap;
   justify or remove it.
8. **Aggressive dual-λ.** `AugmentedLagrangianConfig(dual_lr=5.0, lambda_max=100,
   target_violation=0)` (`lagrangian.py` ~L44-48) drove λ to ~87 in prior runs — over-
   correcting from "never retrain" to "always retrain," which throws away the cost
   objective. Neither extreme is the cost-aware sweet spot. Tune `dual_lr`/`target_violation`.
9. **Scenario "randomization" is a fixed round-robin no-op.** `build_multi_scenario_env_factory`
   claims "random scenario per reset" but uses `counter % len` and leaves `rng` unused
   (`ppo.py` ~L267-287). Confirm and decide if that biases the training curriculum.
10. **Single `DummyVecEnv` (no parallelism)** (`ppo.py` ~L94) — a legitimate speedup
    opportunity (vectorized envs) without changing the science (§3).

For each confirmed item, write a **failing unit test first**, then fix, then show the test
passes. Log each as a `W-` entry with the fix and its verification.

---

## 3. Deliver the audit BEFORE changing anything (§1 of the user's spec)

Inspect and produce a written audit covering: repo structure; training code; env
definitions; reward functions; PPO config; scenario generation; baseline implementations;
evaluation code; Gate A–E definitions; logging/checkpointing; existing experiment outputs;
the Elec2 pipeline; all config files; seed handling; likely bugs; compute bottlenecks;
incomplete experiments; and assumptions not experimentally justified.

Structure it as: **(1) confirmed facts, (2) likely bugs, (3) likely weaknesses,
(4) missing experiments, (5) reproducibility risks, (6) evaluation risks, (7) training
risks, (8) improvement opportunities.** Only then start fixing, root-cause first.

---

## 4. Make the killed run robust — checkpoint / resume / overnight (§2)

Rebuild the experiment driver so the multi-hour run **never restarts from zero**:
- Per-seed lifecycle: `seed → train → periodic checkpoint → evaluate → atomic result write
  → next seed`. Writes must be **atomic** (temp file + rename) so a mid-write kill can't
  corrupt results.
- Periodic model checkpoints + periodic evaluation; persistent experiment state; resume
  from the last completed seed *and* from the last checkpoint within a seed.
- Deterministic init where appropriate; **independent** seeds across runs; graceful
  SIGINT/kill handling that flushes state; automatic structured logging; experiment
  metadata + a full config snapshot per run.
- Support running one seed alone, all 10, or continuing from the last completed seed.
- Fix the terminal-reset issue so a killed TUI leaves the shell usable (disable mouse
  reporting / reset on exit).

---

## 5. Legitimate efficiency only (§3)

Reduce runtime **without changing the science**. Inspect for: unnecessary env resets;
slow Python loops; redundant inference / dataset reloads; recomputation; CPU/GPU
under-utilization; excessive logging; eval/serialization overhead; poor batching;
vectorized-env and parallel-seed opportunities. **Never** cut training steps, drop hard
scenarios, drop seeds, evaluate only favorable checkpoints, or alter test data/metrics to
make results look better. State the measured before/after wall-clock for each optimization.

---

## 6. Repair PPO training — real RL diagnosis, not "more steps" (§4)

Investigate the "never-retrain" collapse properly. Examine: reward scale & sparsity;
action imbalance; clip range; **entropy coefficient**; LR; γ; GAE-λ; batch/minibatch size;
epochs; value-loss coef; **advantage & observation & reward normalization**; rollout
length; network arch/init; exploration; constraint penalties; the augmented-Lagrangian
update, multiplier init/LR/scaling; episode length; terminal rewards; temporal credit
assignment.

Instrument training so you can *see*: action frequencies; retrain frequency; each reward
component (Δf1 vs cost vs SLA-penalty, logged separately); constraint violations; λ
trajectory; policy entropy; value/policy loss; KL; explained variance; episode return;
compute; F1; constraint satisfaction. Then **classify** the agent's behavior as: actually
learning / under-exploring / poorly-scaled rewards / exploiting a loophole / optimizing the
wrong objective / genuinely sensible. Fix the diagnosed cause; do not assume more steps
fixes it.

---

## 7. Contested scenarios done right (§5)

Audit contested-scenario generation. The **training** distribution must contain meaningful
cases where retraining is beneficial, wasteful, better delayed, or better immediate; drift
weak/strong, transient/persistent; compute cheap/expensive; degradation cheap/costly.
Randomize training scenarios broadly; **hold test scenarios out**. Do **not** engineer a
scenario distribution that makes RSO win by construction — a reviewer will see it and so
will you.

---

## 8. Kill data leakage; enforce TRAIN / VAL / TEST (§6, §8)

Audit every path from Elec2 → drift gen → scenario gen → normalization → hyperparameter
tuning → checkpoint selection → test eval. Ensure **test information cannot touch**
training, HP selection, reward design, checkpoint selection, seed selection, or scenario
selection. Formalize the protocol and document what information is allowed at each stage:
- **TRAIN:** optimize PPO on training scenarios.
- **VALIDATION:** tune HPs, select the checkpoint, diagnose behavior.
- **FINAL TEST:** frozen model + frozen HPs + untouched scenarios + 10 independent seeds,
  **no further tuning**. The test set stays untouched until the methodology is frozen.
- Watch specifically for normalization statistics fit on test data, and for scorer/oracle
  info leaking future labels into observations.

---

## 9. Fair, strong baseline suite (§7)

Audit/implement baselines: reactive-full; never-retrain; fixed-period; threshold-based;
drift-triggered; an oracle/lookahead **only if scientifically appropriate** (clearly
labeled as a skyline, not a competitor); and RSO. Every method runs under **identical**
data, prediction model, eval windows, drift conditions, compute accounting, quality
metric, initial conditions, and test scenarios. **Do not weaken baselines.** If a baseline
is strong, keep it; if RSO loses to it, diagnose why.

---

## 10. Complete Gate A properly (§9)

Run the intended experiment: **10 independent seeds × 30,000 training steps × contested
scenarios**, method frozen. Do not claim Gate A success until all required runs finish.
Per seed, persist: seed; config; final checkpoint; best-validation checkpoint; training
curves; eval metrics; compute; F1; retrain count; constraint violations; runtime;
failures/restarts; scenario stats. Report **per-seed values** plus mean/median/std/SE/CIs.
**Never report only the best seed.**

---

## 11. Trade-off surface, not one number (§10)

Report F1, compute, compute-savings, retrain count, carbon proxy, latency (if relevant),
constraint violations, quality degradation, and reward. Produce a **Pareto analysis** and
classify RSO as: (A) dominates, (B) approximately ties, (C) favorable trade-off, (D) loses,
or (E) scenario-dependent. Do not collapse to a single metric if that hides real trade-offs.

---

## 12. Reconsider Gate E scientifically — pre-registered, three claim levels (§11)

Do **not** redefine Gate E to make it pass. Instead ask whether the strict "cheaper AND
equal-or-better F1, zero trade-off" criterion is the right research question. Pre-register
objective statistical criteria (before seeing final-test numbers) for three levels:
- **Claim A (strongest):** RSO reduces compute while maintaining or improving F1.
- **Claim B (trade-off):** RSO achieves substantial compute savings for a small, explicitly
  quantified F1 cost.
- **Claim C (negative):** RSO gives no meaningful advantage over strong baselines.
If A fails but B holds, report B honestly and position the paper/product around it. The
existing Elec2 signal (~69% compute saved for ~1.7% F1) is a Claim-B candidate — see §16.

---

## 13. Statistical rigor — name the unit of analysis (§12)

Fix the pseudoreplication (§2.5). Decide and state the **unit of analysis** (independent
training seed is the honest choice for an RL policy claim; eval-stream realizations are a
second, nested factor). Use paired comparisons across *identical scenario realizations*
where valid; report mean/std/CIs, bootstrap CIs, effect sizes, and significance **only
where the design justifies the test**. Do not call p<0.05 "significant" under a design that
doesn't support it. Avoid pseudoreplication in the reported n.

---

## 14. Ablations that answer "why does it work" (§13)

Prioritize scientifically-important ablations; don't explode the compute budget. At least:
without contested scenarios; without augmented-Lagrangian (fixed-λ); without the constraint
mechanism; with/without entropy/exploration change; alternative reward formulations;
different penalty strengths; key PPO HPs; and possibly the observation-feature set (e.g.
drop the Step-A `sla_margin` / `attribution_concentration` enrichments). Each ablation runs
under the frozen eval protocol.

---

## 15. Robustness — find the failure boundary (§14)

Vary drift magnitude/frequency/duration, compute price, accuracy penalty, retrain latency,
scenario difficulty, seed, initial model state, and noise. Determine where the policy
generalizes and **where it breaks** — a strong paper reports failure boundaries rather than
hiding them.

---

## 16. Re-verify the Elec2 result from raw logs (§15)

Independently **recompute** the ~69%/~1.7% Elec2 claim from raw
`experiments/phase_e_elec2.json` + code — do not trust the summary. Pin down the exact
baseline, exact F1 definition, exact compute definition, eval horizon, retrain policy,
number of runs, whether the comparison is paired, and any preprocessing leakage. Add CIs
and more seeds. If it reproduces, it becomes a headline Claim-B result; if it doesn't,
find out why. Note the known Airlines caveat (hash-encoding masked the drift) — re-encode
(top-K one-hot + numeric Time/Length) before drawing any Airlines conclusion.

---

## 17. Hunt hidden bugs like a hostile reviewer (§16)

Search for and unit-test: action semantics; off-by-one retrain timing (one step early/late);
reward leakage / future information; baseline unfairness; wrong compute accounting; wrong F1
aggregation/averaging; state-reset bugs; normalization leakage; seed reuse; stale weights;
checkpoint mismatch; cross-episode env contamination; eval-on-training-data; accidental
oracle info. Pay special attention to §2.4 (in-sample vs prequential F1) and §2.5 (seed reuse).

## 18. Automated sanity checks that fail loudly (§17)

Add guard tests/asserts that fail if: test data enters training; future info enters
observations; a baseline gets different data; compute accounting goes negative/impossible;
F1 leaves [0,1]; retraining fires outside allowed conditions; seed handling breaks;
checkpoints corrupt; scenario distributions unexpectedly shift; results get overwritten; or
a run terminates silently/early.

---

## 19. One-command experiment (§18)

Provide `python run_experiment.py --config configs/gate_a.yaml` that: validates config;
prints experiment identity (git hash + config hash); trains each seed; checkpoints; resumes
interrupted runs; evaluates; aggregates; generates plots + tables; writes a machine-readable
results file. Support `--seed`, `--resume`, `--evaluate`, `--dry-run`, `--num-seeds`,
`--steps` where sensible.

## 20. Publication-quality output — honest axes only (§19)

Auto-generate: a **main table** (baselines vs RSO: F1, compute, compute-savings, retrain
frequency, quality-delta); a **Pareto plot** (x=compute/cost, y=F1, all methods); a
**per-seed distribution** plot (not just means); **training curves** (reward, F1, compute,
action distribution, constraint violation, entropy); **robustness plots** across
difficulty/drift; and an **ablation table**. No misleading axes or truncations.

## 21. Experiment ledger (§20)

Machine-readable ledger with: experiment ID; git hash; config hash; date; seed; steps;
dataset version; env version; result; status; checkpoint; metrics. Status ∈
`{NOT_STARTED, RUNNING, COMPLETED, FAILED, INVALID, INTERRUPTED}`. Never mark PASS/FAIL for
an unfinished run.

## 22. Reproducibility (§21)

Record Python/package versions, hardware, seeds, config, git hash, dataset version, env
params. Write a README explaining exactly how another researcher reproduces: training,
Gate A, validation, final test, Elec2 eval, plots, and tables.

---

## 23. Two-stage strategy (§24)

- **Stage 1 — cheap diagnostics** (few seeds, short runs): confirm the policy *learns*, the
  action-collapse is fixed, constraints work, reward components behave, and the env is
  valid. Define numeric sanity gates and only proceed if they pass.
- **Stage 2 — expensive confirmatory** (10 seeds × 30k × contested, frozen method). Do not
  modify the method mid-run.

## 24. Pre-register success criteria (§25)

Before the final run, write down objective criteria for: **STRONG SUCCESS** (statistically
credible improvement or A-level result), **PRACTICAL SUCCESS** (compute savings large enough
to matter despite quantified F1 cost — B-level), **INCONCLUSIVE**, and **NO ADVANTAGE**. The
system must be able to *return* any of these, not assume success.

---

## 25. Final deliverable (produce ALL of A–L)

**A. Diagnosis** — exactly what was wrong (root causes, with evidence).
**B. Changes made** — every meaningful code/config change.
**C. Training protocol** — exact final PPO config + rationale for each choice.
**D. Experiment protocol** — exact Gate A–E methodology and the TRAIN/VAL/TEST split.
**E. Commands** — exact commands to run each stage.
**F. Expected runtime** — per-seed and total, on the RTX 4070.
**G. Resume instructions** — exactly how to continue after an interruption.
**H. Validation checklist** — what must pass before the expensive run.
**I. Final-test checklist** — what must be frozen before touching the test set.
**J. Results interpretation** — a decision tree: for each possible outcome, what to conclude.
**K. Paper-ready claims** — defensible wording for strong / practical-trade-off /
   inconclusive / negative outcomes (write all four; the run decides which you use).
**L. Remaining risks** — honestly, what cannot be guaranteed.

## 26. Guardrails recap
Don't cheat (§22): no seed cherry-picking, no post-hoc metric changes, no tuning on test,
no favorable-checkpoint selection, no fabricated measurements, no silent hypothesis
changes. Don't be passive (§23): diagnose and propose legitimate improvements — better
reward shaping/constraints, state representation, exploration, curriculum, PPO stability,
justified action masking, multi-objective/Pareto training, normalization, temporal
features, or a different RL algorithm if PPO is *demonstrably* inappropriate — each tested
under the same frozen protocol.

---

## Start here
1. Read this file, the referenced code, `docs/results.md` (R-4/R-4b/R-Gate-A-smoke/R-Gate-E),
   and `docs/weaknesses.md`. Produce the §3 audit.
2. Verify or falsify each §2 lead with a failing test, fix root causes (reward scaling,
   ent_coef/normalization, in-sample-vs-prequential F1, per-seed policies, replay rng),
   and commit each with its `W-`/`R-` note.
3. Build the resumable pipeline (§4) and land the efficiency wins (§5).
4. Run Stage-1 diagnostics (§23) to a predefined sanity gate; only then freeze the method
   and pre-register success (§24).
5. Run the Stage-2 confirmatory Gate A (§10) + re-verify Elec2 (§16); produce §19–21 outputs
   and the full §25 deliverable.
6. Report the truth, whichever way it lands. Reaching "done" means the experiment is
   rigorous and its conclusion is trustworthy — not that RSO won.
