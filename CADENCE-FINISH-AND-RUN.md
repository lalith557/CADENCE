# CADENCE — Finish & Run Driver (complete what's unfinished, then run the experiment)

> Paste below the divider into a fresh Claude Code session in
> `C:\Users\Lalith Gona\HTML\Capstone Project`. Prior sessions did the fixes and the
> pre-registration but STOPPED before finishing the tooling and before running the
> decisive experiment. Your job: verify what's actually done, finish the bounded
> engineering, then run the science to completion in a way that survives interruption.

---

## 0. Why prior sessions stopped, and the rule that fixes it

The two unfinished tasks are **multi-hour compute jobs** (Stage 1 ≈ 2–4 h; Stage 2 ≈
10× longer) and there is **no resumable runner**, so a session can't finish them in one
sitting and can't hand off. Therefore:

**THE ANTI-STOP RULE:**
- **Bounded coding tasks** (build a file, fix a bug, boot docker, commit): do NOT stop
  until each is actually finished or you are genuinely blocked — then say exactly what
  blocked you.
- **Multi-hour compute** (Stage 1 / Stage 2): do NOT block on it and do NOT silently
  stop. Launch it as a **background, checkpointed, resumable** job, then report: what's
  running, the log path, current progress, and the **exact resume command** the next
  session (or the user) runs to continue. "It's a long job" is never a reason to leave
  it un-launched or un-resumable.

Work top to bottom. Commit after each numbered section. Keep `docs/results.md`,
`docs/weaknesses.md`, and the ledger current.

---

## 1. FIRST — self-verify against the plan (don't trust commit messages)

Read `CADENCE-GATE-A-MASTER-PROMPT.md`, `docs/gate_a_preregistration.md`,
`docs/gate_a_audit.md`, and the tail of `docs/results.md`. Then, **from the actual repo
state**, produce a DONE / PARTIAL / NOT-DONE table covering every section of the master
prompt (§1–§26) and the pre-registration's Stage-1/Stage-2 runs. A task is DONE only if
the code exists AND (for experiments) a real result entry + artifact exists. Known state
to confirm or correct:
- DONE: §3 audit; §2 fixes W-32..W-37 (reward rescale, ent_coef, VecNormalize,
  prequential reward, per-seed policies, replay RNG); §24 pre-registration; paper-scale
  H1 (R-Gate-B-n10) and H3 (R-Gate-F-mnist-n10).
- NOT DONE: Stage-1 diagnostics run; Stage-2 Gate A run; §16 Elec2 re-verification;
  §19 `run_experiment.py`; §4 checkpoint/resume; §20/§21 experiment ledger; docker
  booted; dashboard real-CDAG (W-30). Uncommitted: `cadence/rso/env.py`,
  `tests/unit/test_rso_audit.py`; untracked: `configs/gate_a.yaml`.

Write this table to `docs/gate_a_status.md` so progress is trackable across sessions.

---

## 2. Commit the dangling work first (nothing lost)

`git add -A` the in-progress `cadence/rso/env.py`, `tests/unit/test_rso_audit.py`, and
`configs/gate_a.yaml`; run the test suite; commit (`chore: commit in-progress rso/env +
gate_a config before finishing Gate A`). Do this before anything else.

---

## 3. Build the resumable runner + ledger (§4, §19, §20, §21) — BOUNDED, finish it

Create `run_experiment.py` (repo root) — one command that survives laptop interruption:
- `python run_experiment.py --config configs/gate_a.yaml` runs the full protocol.
- Per-seed lifecycle: `seed → train → periodic checkpoint → evaluate → ATOMIC result
  write (temp file + os.replace) → next seed`.
- **Checkpoint/resume:** each seed's PPO checkpoint + a per-seed result JSON written as it
  completes; on restart, skip seeds already `COMPLETED`, resume the in-flight one from its
  last checkpoint. A kill must never cost more than one seed.
- **Experiment ledger** `experiments/gate_a_ledger.json`: per (stage, seed) rows with
  `status ∈ {NOT_STARTED, RUNNING, COMPLETED, FAILED, INVALID, INTERRUPTED}`, git hash,
  config hash, timestamps, checkpoint path, metrics. **A killed/unfinished seed is
  `INTERRUPTED`, never PASS/FAIL.**
- Flags: `--stage {1,2}`, `--seed N` (one seed), `--resume` (continue from ledger),
  `--num-seeds`, `--steps`, `--dry-run` (validate config + print plan, no training),
  `--evaluate` (eval existing checkpoints only).
- Graceful SIGINT: flush ledger + partial results, print the exact resume command, exit.
- Reuse the existing `benchmarks/phase_a_run.py` internals (env factory, PPO, baselines,
  the W-32..W-37 fixes) — wrap, don't rewrite. Confirm `--per-seed-policies` is ON.
- Add a `--backend {local,colab}` note or a `scripts/colab_gate_a.py` hook so Stage 2 can
  be handed to Colab with the identical config if local wall-time is too long.

Add a unit test that kills a 2-seed dry-run mid-way and asserts `--resume` continues from
seed 2, not seed 0. Commit.

---

## 4. Run STAGE 1 (diagnostics) — resumable, then gate on the pre-registration (§23)

Protocol is frozen in `docs/gate_a_preregistration.md` Part 1: **3 seeds × 3,000
timesteps × 3 contested scenarios**, six fixes on, per-seed policies. Launch it
**in the background** with logging:

```bash
python run_experiment.py --config configs/gate_a.yaml --stage 1 --resume
```

While it runs, do NOT stop — monitor the log and the ledger. When it finishes, evaluate
**all 8 Stage-1 gates (G1–G8)** exactly as written (no action collapse, entropy ≥0.15,
reward increases, cost signal visible ≥1% of Δf1, not SLA-gaming, λ≤50, real cross-seed
variance, finishes in budget). Write the outcome to `docs/results.md` as
`R-Gate-A-stage1` with the per-gate PASS/FAIL and the instrumentation (action histogram,
entropy, λ trajectory, reward-component decomposition).

- **If all 8 gates pass:** freeze the method and go to §5.
- **If any gate fails:** log `R-Gate-A-stage1-fail-<n>`, diagnose that ONE gate, propose a
  single fix, write a failing test first, fix, and **restart Stage 1 clean** (per the
  pre-registration's single-fix loop). Do NOT proceed to Stage 2 by patching and pushing on.

---

## 5. Run STAGE 2 (confirmatory Gate A) — overnight/background/Colab, resumable (§10)

Only after Stage 1 passes. Protocol frozen: **10 seeds × 30,000 timesteps × 8 scenarios**
(5 default + 3 contested), baselines periodic + reactive-full + no-op, no further tuning.

```bash
python run_experiment.py --config configs/gate_a.yaml --stage 2 --resume
```

This is the long one. Launch it in the background (or hand to Colab via
`scripts/colab_gate_a.py` with the same config) and report the resume command. Each
session that reopens should run the same `--resume` line to advance more seeds until the
ledger shows all 10 `COMPLETED`. Then aggregate: per-seed values + mean/median/std/SE/CIs,
paired Wilcoxon on `(scenario, policy_seed)` with Bonferroni, and the **Pareto trade-off
surface** (F1 vs compute, all methods). Apply the pre-registered verdict criteria
(STRONG / PRACTICAL / INCONCLUSIVE / NO-ADVANTAGE) and pick the **most conservative** that
fits. Write `R-Gate-A-final` with ALL 10 seeds, the verdict, and the §25 A–L deliverable.

---

## 6. Re-verify Elec2 from raw (§16) — BOUNDED

Independently recompute the ~69% compute / ~1.7% F1 Elec2 claim from
`experiments/phase_e_elec2.json` + code (exact baseline, F1 def, compute def, horizon,
paired-ness, leakage check); add CIs and more seeds. Write `R-Gate-E-verified`. If it
doesn't reproduce, say why. This is the Claim-B fallback headline — it must be audit-proof.

---

## 7. Close the product-launch gaps (§ product) — BOUNDED, finish them
- **Boot docker for real** (W-31/§P3): `docker-compose up --build dashboard mlflow`,
  confirm `localhost:8501` renders, capture a log/screenshot. If the daemon is off, say so
  and leave the exact command; don't mark it done on config-validation alone.
- **Real dashboard CDAG** (W-30): extend the experiment artifacts with raw NOTEARS weights
  + real responsibility + real decision + real cost-saved, and render the true graph — no
  illustrative panels.
- **Verify the API** end-to-end (`/score`, `/monitor`, `/decide`, `/health`) against the
  learned policy; add/confirm tests.
- **CI green** on a real push (W-8); remove `mypy continue-on-error` (W-11).

---

## 8. Definition of done + what to report every session
Done when: `run_experiment.py` + ledger exist and survive a kill; Stage 1 ran and its 8
gates are logged; Stage 2 ran to 10/10 `COMPLETED` with a pre-registered verdict in
`R-Gate-A-final`; Elec2 re-verified; docker booted + dashboard real + API verified + CI
green; weakness register converged; `docs/gate_a_status.md` reflects reality.

**End every session with a status report:** what finished, what's running (with log path +
progress), the exact `--resume` command to continue, and what's blocked and why. Never end
a session silently — if a long job is mid-flight, that's expected; say so and leave the
resume command. If a bounded task is unfinished, that's not expected — finish it or name
the blocker.

## 9. Honesty (binding)
The pre-registration is a contract: no seed cherry-picking, no post-hoc metric changes, no
tuning on the final test, no checkpoint-by-test-score, no fabricated numbers, no silent
hypothesis change. A killed run is `INTERRUPTED`, not FAIL. If Stage 2 lands NO-ADVANTAGE,
reframe honestly around H1-AUROC / H3 / the surrogate per the pre-registration — that's a
real result, not a failure to hide.

## Start here
1. §1 status table → §2 commit dangling work → §3 build the resumable runner (finish it).
2. §4 launch Stage 1 in background, gate it. Then (if pass) §5 launch Stage 2 resumable.
3. In parallel with the long runs, finish the bounded §6 and §7 items.
4. Report status + resume command at the end of the session. Don't stop on bounded tasks;
   don't block on long ones — background + checkpoint them.
