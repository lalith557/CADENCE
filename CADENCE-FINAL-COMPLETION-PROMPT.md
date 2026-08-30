# CADENCE — Final Completion Prompt (Weakness Closure → Main-Track + Launch)

> Paste everything below the divider into a fresh Claude Code session opened in
> `C:\Users\Lalith Gona\HTML\Capstone Project`. This is the **final push**: close
> every open weakness, prove every anchor at full statistical scale, and take the
> system from "honest prototype" to "launchable + main-track-submittable." It
> continues an existing, working build — read the current state before changing it.

---

## 0. THE HONESTY CONTRACT (read first — this overrides every other instruction)

You are finishing a real research + product system. The value of everything you do
here depends on one rule:

**Never fabricate, cherry-pick, hardcode, or overstate a result. Ever.**

- Every number that lands in `docs/results.md`, the paper skeleton, the dashboard, or
  `docs/product.md` must come from a run you actually executed, at the stated scale,
  reproducible from the stated command.
- "Prove each anchor" means **run the full-scale experiment and report what it shows** —
  not "make the table say PASS." If an anchor holds, prove it rigorously. If, after an
  honest full-scale run, an anchor does **not** hold, you write that down as a negative
  result and **reframe the paper's claim to the strongest thing that is actually true**.
  A reframed honest paper is publishable; a faked one ends a career.
- Do not lower an SLA, shrink a baseline, drop unfavourable seeds, or pick the one
  scenario that wins in order to manufacture a pass. If you catch yourself tuning the
  *evaluation* to get a nicer number, stop and log it as a threat to validity.
- The current ledger is honest (it records its own negatives). Keep it that way. That
  honesty is the project's most valuable asset — protect it.

If you cannot make a claim true, your job is to make the paper/product tell the truth
well — not to make the claim look true.

---

## 1. Source of truth & current state

Read before touching anything:
- `CADENCE-master-reference.md`, `CADENCE-datasets-verified.md`, `CADENCE-BUILD-PROMPT.md`,
  `CADENCE-EXECUTION-PLAN.md` (all still in force), and the current `docs/`
  (`decisions.md`, `tech.md`, `results.md`, `weaknesses.md`, `phase-plan.md`,
  `product.md`, `prior_art.md`, `paper/`).
- **Where the project actually stands** (verified, not commit-message claims):
  - Every component is **built and wired**; the pipeline runs end-to-end; GPU is real.
  - **H3** is the only clean positive (Split-MNIST, p=0.031, **n=5**). H3 on Fraud is a
    logged **negative** (single-task) — keep it, it's a correct finding.
  - **H1** (GNN>PSI) is a **small-n** positive (9 samples, hard-subset only; graph
    recovery SHD=1.0 is poor).
  - **H2** (the RSO cost-win — the *flagship* Claim C) is **NOT statistically proven**:
    Gate A was killed before stats; real-drift Gate E returns "FAIL on strict criterion,
    Pareto-not-dominant."
  - Generality is **plumbing-deep only** (SLAs were set so CADENCE never had to act).
  - Product surface is a **demo shell**: dashboard CDAG panel is illustrative, Docker was
    never booted, REST API is commented out, CI never ran.
  - **27 weaknesses open**; the register never converged. The prior-art search (W-4) was
    never done — this alone blocks a main-track submission.
- Keep the four living docs current in the **same commit** as the code. Commit at every
  gate. First: clean up stale weakness entries (W-22, W-23, W-28 are implemented in code
  now — mark them fixed with the commit/R-entry that closed them; don't leave the register
  lying).

---

## 2. What "100% / done" means — three acceptance bars, all required

You are done only when **all three** are true and demonstrated:

**A. RESEARCH bar (main-track-submittable)**
- H1, H2, H3 each run at the paper protocol: **≥10 seeds × ≥5 scenarios** (where
  applicable), paired Wilcoxon/t-test with p-values, effect sizes, 95% CIs.
- Each anchor claim (A/B/C) has a verdict backed by an adequately-powered run:
  SUPPORTED (with stats) or, honestly, NOT-SUPPORTED-and-reframed.
- The **prior-art search is done** and `docs/prior_art.md` is a real related-work section
  that positions all three claims and narrows any that collide.
- At least **one adequately-powered, strictly-defensible positive headline result** exists.
- The paper skeleton is populated from real results and is submission-shaped
  (abstract, related work, method, experiments, honest limitations, reproducibility).

**B. PRODUCT bar (launchable prototype)**
- Clean-clone quickstart works **from scratch** (fresh venv or Docker) — verified by
  actually running it, with a captured screenshot/log, not "config-validated."
- The dashboard shows **real** data (real NOTEARS graph, real responsibility scores, real
  decision + real cost saved) — no illustrative/placeholder panels.
- A working **REST API** (FastAPI) exposes score + monitor + decide endpoints.
- **Docker/compose actually boots**; **CI is green** on a real push; packaging installs.
- `docs/product.md` states, honestly, what ships, what it does, and the ROI story grounded
  in the *real* (Pareto) results — no claims the experiments don't support.

**C. CONJUNCTION bar (every part works together)**
- The **learned 15-d PPO policy** (not the rule fallback) drives the executor in Gate C,
  Gate E, the API, and the dashboard — threaded through and verified.
- One command runs the **entire loop end-to-end with the learned policy** on both a
  synthetic and a real dataset, exercising every §4 fallback, and it passes.
- Full test suite green; a new **end-to-end integration test** asserts the whole chain;
  no component silently falls back or no-ops by accident.
- The weakness register **converges**: every W-# is fixed, or explicitly deferred with a
  one-line justification and no open blocker/major remains.

---

## 3. Compute discipline (so long runs finish and nothing is lost)

The paper-grade runs are heavy — the single-seed 800-step Gate-A run already took ~2.75h;
a 10-seed × 30k-step × multi-scenario sweep is a multi-hour-to-multi-day job. So:
- **Profile and speed up the sandbox first.** Most wall-time is in `_do_partial_retrain`'s
  FraudNet epochs, not PPO. Cut eval-fidelity retrains to the minimum that keeps results
  valid; vectorize; keep everything on GPU; cache Fisher/pretrain.
- **Checkpoint every long run** to a per-seed/per-cell JSON so a kill leaves usable partial
  results (the W-26 pattern). Never run a multi-hour job that produces nothing if
  interrupted. Resume-able > monolithic.
- **Run the big sweeps in the background** with logs, and write the `R-` entry only after
  the summary is actually produced. If a run is killed, say so and record what was
  collected — do not infer the missing numbers.
- If a sweep genuinely exceeds this machine, **prepare a Colab/Kaggle notebook** that runs
  the identical config and pulls results back — flag it, don't silently skip.
- Fix the terminal-reset issue so a killed TUI doesn't corrupt the shell (exit cleanly).

---

## 4. WORKSTREAM R — prove every anchor at paper scale

Do these in order; each ends in a real `R-` entry + commit. Report honest outcomes.

**R1 — Prior-art search (W-4). Do this first; it can reshape the claims.**
Semantic Scholar / Google Scholar sweep on: causal graphs for model/drift debugging;
activation-space causal analysis; RL for retraining/MLOps decisions; cost/carbon-aware
continual learning; surrogate counterfactual effect estimation. Write `docs/prior_art.md`
as a real related-work section. If any hit collides with Claim A/B/C, **narrow the claim**
and note it in `decisions.md`.

**R2 — Close H2, the flagship (Claims C). This is the make-or-break.**
Run the full Gate A: **≥10 seeds × the 5 default + 3 contested scenarios × 30k PPO
timesteps**, augmented-Lagrangian dual-λ, EWC-in-sandbox partial, learned 15-d obs.
Compare RSO vs periodic vs reactive-full on compute/carbon and SLA-recovery, paired
Wilcoxon, CIs. Then re-run **real-drift Gate E at ≥10 seeds** with a *proper* Airlines
encoder (one-hot top-K airlines + numeric Time/Length, not hashing) so the drift signal
survives. Verdict, honestly:
- If RSO gives a **Pareto/strict** win on contested + real drift → H2 SUPPORTED; this is
  the headline.
- If it only gives a **controllable trade-off** → say so; reframe the paper's core claim
  to "a tunable cost/quality policy with causal attribution," and lean the headline on the
  strongest supported claim (likely H3/H1) instead. Either way, no faking.

**R3 — Re-run H1 at paper scale (Claim A).**
5 scenarios × ≥10 seeds (≥50 paired samples); report **easy and hard subsets separately**.
Address the SHD=1.0 graph-recovery weakness: loosen NOTEARS L1 / increase window count so
the recovered graph is non-degenerate, or report graph-recovery as an explicit limitation
while keeping the attribution claim. Fix the `causallearn` PC `math domain error` dense
fallback or document it as a known limitation.

**R4 — Harden H3 (Claim, forgetting).**
Re-run Split-MNIST at **≥10 seeds**; add a **fair full-retrain-with-replay** baseline as
an ablation (the current naive-no-replay full is the weakest baseline); λ_ewc sweep on both
MNIST and Fraud to show the protect-vs-adapt knee. Keep the honest Fraud negative framed as
the single-task vs multi-task finding.

**R5 — Real generality (not plumbing).**
Re-run the **tree adapter at SLA≈0.45** (just below baseline) so CADENCE must actually act,
and report whether it beats baselines when the decision is live. Do a **real text-drift**
experiment (W-29): either Amazon Reviews 2023 or a deliberate domain-shift Yelp split that
produces measurable degradation — so the text result tests drift, not just wiring.

**R6 — Reward-weight calibration (W-7) & stats hygiene (W-5, W-20, W-25).**
Report the (w_gpu, w_co2, λ) sensitivity sweep and pick from the stable center; ensure every
headline is ≥10 seeds with non-degenerate seed variance once policies are learned.

---

## 5. WORKSTREAM P — make it a launchable prototype

**P1 — Real dashboard (W-30).** Extend `experiments/*.json` to carry raw NOTEARS weights +
real per-node responsibility + real decision + real cost-saved per (scenario, seed); make
`dashboard/app.py` render the true CDAG and true numbers. No illustrative panels remain.

**P2 — REST API.** Implement the commented-out FastAPI service: `/score` (run a model
through the loop), `/monitor` (drift status + attribution), `/decide` (RSO action + cost),
`/health`. Wire it to the learned policy. Add API tests.

**P3 — Boot the stack for real (W-31, W-9).** `docker-compose up --build` the dashboard +
MLflow (+ API); verify from a clean clone; capture a screenshot/log at `localhost:8501`.
Confirm the Python-3.10 container matches the 3.13 dev behaviour.

**P4 — CI + packaging (W-8, W-11, W-12).** Push to GitHub; get CI green on Ubuntu; add the
Docker-build job; remove `mypy continue-on-error` and fix the type errors; add pre-commit
(ruff+format+mypy). `pip install -e .` + a 5-minute quickstart works from a fresh clone.

**P5 — Ops hygiene (W-2, W-13, W-14, W-15).** `.gitignore` datasets; `.env.example` +
`docs/secrets.md`; migrate MLflow to `sqlite:///experiments/mlflow.db` (drop the file-store
workaround); de-duplicate the config YAMLs behind a base + overrides.

**P6 — Honest product doc.** `docs/product.md`: what ships, the differentiator ("why it
broke," shown from the real dashboard), and an ROI story grounded only in measured results.

---

## 6. WORKSTREAM C — everything works in conjunction

**C1 — Thread the learned policy everywhere.** Gate C, Gate E, the API, and the dashboard
must consume `experiments/rso_ppo_phase_a.zip` (or its ≥10-seed successor), not the rule
fallback. Verify no `ppo_obs_dim_mismatch` / `using_rule` warnings on the intended path.
The rule fallback stays as a genuine safety net, never as the default.

**C2 — End-to-end integration test.** One command + one `pytest` integration test drives:
real/synthetic data → drift → PSI trigger → CDAG → GNN → surrogate → **learned RSO** →
executor (partial/full) → shadow/canary → promote/rollback → escalation. Assert every stage
fires and every §4 fallback has a test that forces it.

**C3 — Register convergence + doc truth.** Close every open W-# or explicitly defer it with
justification; mark the stale-but-done ones fixed; ensure `results.md`, `product.md`, and
the paper skeleton agree with each other and with the code. No blocker/major left open.

---

## 7. Definition of done — the final checklist (all must be checkable)

- [ ] Prior-art search done; `prior_art.md` real; claims narrowed where needed.
- [ ] H1, H2, H3 each run at ≥10 seeds × ≥5 scenarios with paired stats + CIs; verdicts honest.
- [ ] ≥1 adequately-powered, strictly-defensible positive headline result.
- [ ] Paper skeleton submission-shaped, populated from real results, with a real limitations section.
- [ ] Learned PPO drives executor/API/dashboard; verified end-to-end.
- [ ] Clean-clone quickstart actually run (screenshot/log); Docker booted; CI green.
- [ ] Dashboard 100% real data; REST API live + tested.
- [ ] End-to-end integration test green; all 8 fallbacks tested; full suite green.
- [ ] Weakness register converged (no open blocker/major); living docs mutually consistent.
- [ ] `docs/STATUS.md` reflects the true final state, honestly.

## 8. Order of operations
R1 (prior-art) → P5/P4 hygiene + speed-up §3 → **R2 (H2, the flagship)** → R3 (H1) →
R4 (H3) → R5 (generality/text) → R6 (calibration) → P1–P3, P6 (product) → C1–C3
(conjunction) → final adversarial audit + `STATUS.md`. Gate each; commit each; run the
weakness-audit loop after each; keep the four docs live.

## 9. Start here
1. Read the reference docs + current `docs/` state; correct the stale weakness entries.
2. Do R1 (prior-art) and the §3 compute speed-ups first — they de-risk everything after.
3. Then work the order in §8, one gate at a time, checkpointing long runs, committing at
   each gate, and writing each `R-` entry only from a run that actually finished.
4. When an anchor holds, prove it to the hilt. When one doesn't, tell the truth and reframe.
   Reaching "100%" means the project is *complete and honest*, not that every hypothesis won.
