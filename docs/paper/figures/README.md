# Paper figures — provenance and captions

All figures here are auto-generated from **measured** experiment artifacts under
`experiments/*.json`. No numbers are typed into the source; regenerating from the
latest artifacts is a single command:

```
python scripts/make_paper_figures.py
```

Both `.png` (for slides / MD previews) and `.pdf` (for LaTeX) are produced at
300 dpi with `tight_layout`.

## Current headline figures (2026-09-03 refresh)

| File | Source artifact | Backs |
|---|---|---|
| `fig_elec2_pareto.{png,pdf}` | `experiments/phase_e_elec2.json` (R-Gate-E-verified) | Claim B (PRACTICAL) — the ~69% GPU-hr saving for a ~-0.017 F1 gap Pareto frontier |
| `fig_h1_subset.{png,pdf}` | `experiments/phase_b_n10.json` (R-Gate-B-n10-subset) | H1 — GNN attribution wins on the HARD subset (MRR/AUROC, p ≤ 0.032 Bonferroni) |
| `fig_h3_mnist_forget.{png,pdf}` | `experiments/phase_f_mnist_n10.json` (R-Gate-F-mnist-n10) | H3 — Partial-EWC beats naive-full AND full-with-replay on Task-A forgetting |

## Legacy figures (Aug 31, from earlier sessions)

Kept for reference; likely superseded by the ones above and by later runs.

| File | Notes |
|---|---|
| `elec2_f1_over_time.png` | Per-window F1 trajectory on Elec2; a time-series companion to the Pareto plot. |
| `false_alarm_vs_true_alert.png` | PSI-trigger specificity/sensitivity (R-Gate-F robustness). |
| `h1_scorer_comparison.png` | Predecessor of `fig_h1_subset` — all-scenarios only, no EASY/HARD split. |
| `h3_split_mnist_forgetting.png` | Predecessor of `fig_h3_mnist_forget` — 2-arm version (no full-with-replay). |
| `robustness_edge_dropout.png` | GNN AUROC vs CDAG edge-drop fraction. |
| `surrogate_calibration.png` | R-Gate-C surrogate predicted-vs-measured F1 calibration. |
| `tree_pareto_frontier.png` | LightGBM adapter (R-Gate-F-tree-sla045) Pareto plot. |

## Suggested paper captions

**Fig 1 — `fig_elec2_pareto`:**
> "Real-drift end-to-end on Elec2 (river). Each point is one seed (n=3);
> markers with `+` show per-strategy means. CADENCE's rule policy sits on the
> Pareto frontier: it saves **69.1%** of the GPU-hours the reactive-full baseline
> spends, for a **−0.017 F1** gap (95% bootstrap CI [−0.025, −0.009])."

**Fig 2 — `fig_h1_subset`:**
> "H1 attribution at paper scale (n=10 seeds × 5 scenarios). Bars are means
> ± std; solid = EASY subset (3 PSI-saturated scenarios), hatched = HARD
> subset (2 contested scenarios — `time_gradual`, `v14_concept_shift`). On
> HARD, the CDAG+GNN scorer beats both baselines on MRR (p<0.001 vs PSI,
> p=0.032 vs structural) and AUROC (same). On EASY, PSI is already saturated
> at 1.0."

**Fig 3 — `fig_h3_mnist_forget`:**
> "H3 — Split-MNIST Task-A forgetting under three retrain arms (n=10 seeds).
> Boxes: IQR; dashed line: mean. Partial EWC (ours) preserves Task A almost
> perfectly (median ≈ −0.03, small variance) while full-retrain-naive
> loses ~0.40 F1 points on Task A. Even under the fair full-with-replay
> baseline, partial EWC ties or slightly leads. Paired Wilcoxon
> `partial < full`: p<0.001; `partial < full_with_replay`: also significant."
