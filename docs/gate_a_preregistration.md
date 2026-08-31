# Gate A — Pre-registration (§23 Stage-1 gates + §24 Stage-2 criteria)

**Signed off:** 2026-09-01, before any Stage-1 or Stage-2 run.
**Anchoring commit:** `cfcf860` (all W-32..W-37 fixes landed).
**Prime directive:** these criteria are frozen. If a run doesn't meet the
Stage-1 gates, we DO NOT proceed to Stage 2 — we diagnose. If Stage 2 lands
between two verdicts, we pick the *worse* one (never the more flattering
interpretation).

---

## Part 1 — Stage-1 sanity gates (§23)

**Stage-1 protocol.** 3 seeds × 3,000 PPO timesteps × 3 contested scenarios
under the six post-audit fixes (reward rescaled, `ent_coef=0.01`,
`VecNormalize`, prequential reward, `--per-seed-policies`, per-env replay
RNG). No baselines. Purpose: confirm the *policy actually learns* under the
post-audit reward before we spend the Stage-2 compute.

**Sanity gates. All must pass to proceed to Stage 2.**

| # | Gate | Numeric criterion | Rationale |
|---|---|---|---|
| G1 | **No action collapse** | Per-seed action histogram: none of `{no-op, partial, full}` accounts for more than **85 %** of steps averaged over the last 30 % of training. | The R-4b bang-bang pathology. If the policy still picks one action ≥85 % of the time after all six fixes, ent_coef isn't doing its job — diagnose before scaling up. |
| G2 | **Non-trivial policy entropy** | Mean policy entropy over the last 30 % of training ≥ **0.15 nats** (chance = ln(3) ≈ 1.10). | Zero entropy = deterministic collapse. |
| G3 | **Reward increases** | Rolling mean episode return over last 30 % strictly greater than mean over first 30 %. Improvement ≥ **0.10** in raw units, and not driven only by the SLA term (see G5). | Basic learning check — if reward doesn't move, PPO isn't training. |
| G4 | **Cost signal visible** | Absolute value of the cost-penalty term averaged across training episodes ≥ **1 %** of the absolute value of the Δf1 term. | The W-32 fix's headline claim. Verifiable directly from the per-episode reward decomposition we log. |
| G5 | **Reward is not just SLA gaming** | If `sla_penalty_component` shrinks over training AND `cost_penalty_component` also shrinks (RSO learned to save cost while satisfying SLA), G5 passes. If cost penalty grows monotonically = RSO learned to always retrain, which G5 fails. | Rules out the degenerate "λ pushed us to always-retrain" outcome. |
| G6 | **Dual-λ stability** | λ trajectory does not saturate at `lambda_max=100` in the last 30 %. Final λ ≤ **50**. | Prior R-Gate-A-w28 hit 87 near the cap — with the reward rescaled, λ should not need to explode. |
| G7 | **Cross-seed variance is real** | Std of final-30 %-mean reward across 3 seeds ≥ **0.02**. | W-25/W-36: legitimate seed variance under `--per-seed-policies`. |
| G8 | **Every stage finishes** | 3 seeds × 3,000 timesteps completes within 4 hours wall on the RTX 4070 Laptop, no crashes. | Practical — if Stage 1 doesn't finish, Stage 2 (~10× longer per seed) can't. |

**If a gate fails.** Log the failure to `docs/results.md` as
`R-Gate-A-stage1-fail-<n>`, diagnose the specific gate, propose a single
new fix, verify it with a failing test first, and re-run Stage 1. Do NOT
proceed to Stage 2 by "just fixing that one thing and running." Each
diagnostic loop restarts Stage 1 from clean.

**Compute budget for Stage 1:** projected ~2–4 hours on this laptop
(3 seeds × 3,000 timesteps + no paper-fidelity baselines). If projection
proves wrong, Stage 1 gets a hard 6-hour timeout and is retried once
after diagnosis.

---

## Part 2 — Stage-2 outcome pre-registration (§24)

**Stage-2 protocol (frozen).** 10 seeds × 30,000 PPO timesteps × 8 scenarios
(5 default + 3 contested) under the frozen post-audit code. Baselines:
periodic (every N windows), reactive-full (PSI+full), and no-op (never
retrain — added as skyline for cost, floor for F1). No further tuning.

**Unit of analysis.** Each independent training seed produces one policy;
each policy runs *deterministic* eval over the 8 test scenarios (fresh
eval-stream sampling seeded from the policy seed). The paired unit is
`(scenario, policy_seed)` — n = 8 × 10 = 80 paired samples per baseline
comparison. Paired Wilcoxon signed-rank test where the design justifies
it; Bonferroni over all four (baseline × metric) comparisons.

**Primary metrics** (all reported; no single-number claim):

- `mean_F1` per scenario × seed
- `total_gpu_hr` per scenario × seed
- `compute_savings_pct` = `1 − rso_gpu_hr / baseline_gpu_hr`
- `retrain_count` per scenario × seed
- `sla_windows_below` per scenario × seed
- `kg_co2` per scenario × seed

**Four outcome verdicts. Only one applies. Pick the most conservative
that fits.**

### STRONG (Claim A) — RSO strictly dominates on cost at equal-or-better F1

All of:
- Median `compute_savings_pct` vs `reactive_full` ≥ **50 %** across the
  80 paired samples.
- Median `mean_F1` gap vs `reactive_full` ≥ **−0.005** (RSO's F1 is
  within half a percentage point below or above).
- Paired Wilcoxon `rso_gpu_hr < reactive_full_gpu_hr`, one-sided,
  **p < 0.0125** (Bonferroni over 4 baseline×metric pairs at α=0.05).
- Paired Wilcoxon `rso_mean_F1 ≥ reactive_full_mean_F1`, one-sided,
  **p > 0.05** OR positive (RSO is not statistically worse).
- Beats `periodic` on total-cost-at-F1 by the same criteria.
- **Beats no-op on F1** with `p < 0.0125` (RSO isn't just no-op in disguise).

### PRACTICAL (Claim B) — RSO gives a substantial compute win at a small, quantified F1 cost

All of:
- Median `compute_savings_pct` vs `reactive_full` ≥ **50 %**.
- Median `mean_F1` gap vs `reactive_full` in the range **[−0.03, −0.005]**.
- Paired Wilcoxon `rso_gpu_hr < reactive_full_gpu_hr` p < 0.0125.
- `mean_F1` gap statistically significant BUT small (effect size |d| < 0.5).
- Beats no-op on F1 with p < 0.0125.

This is the R-Gate-E Elec2 result's home. **A honest Claim-B result is a
publishable finding** — it says "here is a real, quantified cost/quality
knob no baseline offers."

### INCONCLUSIVE — statistically underpowered or contradictory

Any of:
- Wilcoxon p-values straddle Bonferroni α (some scenarios significant,
  others not, with no clean story).
- Median compute savings between 10 % and 50 % — non-trivial but not
  paper-headline.
- Scenario-conditional wins that don't generalize across the 8 scenarios.
- Any of the primary metrics has cross-seed CV > 100 %.

In this case: report the trade-off surface honestly, run the §14 ablations
to try to identify a contested subset where a claim holds, and if none
holds after ablation, degrade to NO-ADVANTAGE.

### NO-ADVANTAGE (Claim C) — RSO offers no meaningful benefit over strong baselines

Any of:
- Median `compute_savings_pct` vs `reactive_full` < 10 %.
- OR `mean_F1` gap vs `reactive_full` < −0.03 (RSO's F1 is 3+ points below).
- OR RSO loses to no-op on F1 (RSO retraining is *net-harmful*).

In this case: **the honest paper reframes around the components that DID
work** (Claim A — GNN attribution, Claim B — surrogate, H3 forgetting on
multi-task) and reports Claim C as a substantive scientific finding
("cost-aware RL scope selection does not beat well-tuned baselines under
the tested reward + protocol"). This is a legitimate publishable outcome
and it does not invalidate the rest of the system.

---

## Part 3 — Anti-cheating guarantees

Written down so any future me is bound by them:

1. **No seed cherry-picking.** If Stage 2 completes, ALL 10 seeds' results
   land in the results.md entry. If a seed crashed, its status is
   `INTERRUPTED` and the aggregate uses n=9; we do NOT restart just that
   seed with a different init.
2. **No post-hoc metric changes.** The primary metrics list above is
   frozen. If the paper later cites a different metric, that metric must
   have a paper-appendix table showing it was pre-registered here.
3. **No re-running with tweaked hyperparameters.** If Stage 2 lands at
   INCONCLUSIVE and I want to try a smaller `dual_lr`, that becomes a
   NEW pre-registration + Stage 1 + Stage 2, not a "quick rerun."
4. **No checkpoint cherry-picking.** Eval uses each seed's final
   checkpoint (not the best-validation one, since we have no separate
   validation set within a run). Best-validation selection requires a
   third scenario set never seen during training or Stage 2 eval, which
   we don't have — using a val split of the training scenarios would
   leak.
5. **The test scenarios stay untouched until Stage 1 gates pass.** The
   Stage 1 runner uses the 3 contested scenarios only; the 5 default
   scenarios are held out for Stage 2 alone.
6. **If Stage 2 disagrees with the smoke-scale prior results** (e.g.
   R-Gate-E's Elec2 win doesn't generalize to Fraud), that disagreement
   is the finding — not something to explain away.

---

## Part 4 — Decision-tree summary

```
Stage 1 runs → all 8 gates pass?
   │
   ├── YES → freeze method, hand off Stage 2 to Colab
   │           │
   │           └── Stage 2 completes → apply Part-2 verdict criteria
   │                                       ├── STRONG   → paper headline
   │                                       ├── PRACTICAL → paper headline (B)
   │                                       ├── INCONCLUSIVE → ablate then
   │                                       │                    reclassify
   │                                       └── NO-ADVANTAGE → reframe honestly
   │
   └── NO  → log Stage-1 failure, diagnose one gate, single-fix loop,
             restart Stage 1. Do NOT push to Stage 2.
```

---

*This document is the contract. Every subsequent Gate A commit references
back to it. If you want to change a criterion in Part 2, that is a new
pre-registration — do not silently overwrite this file.*
