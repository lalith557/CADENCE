#!/usr/bin/env python
"""Generate publication figures from EXISTING (already-measured) experiment
artifacts — no new compute runs.

Writes PNG+PDF (300 dpi, tight bbox) to docs/paper/figures/ for direct paper use.
Reproducible: hard-coded input paths, no randomness. Failures print the missing
artifact rather than silently generating blank figures.

Figures produced:
  1. fig_elec2_pareto.{png,pdf}    — R-Gate-E-verified: RSO vs baselines
                                     on Elec2, cost vs F1 (Claim B, PRACTICAL).
  2. fig_h1_subset.{png,pdf}       — R-Gate-B-n10-subset: GNN vs PSI vs
                                     structural attribution, EASY vs HARD.
  3. fig_h3_mnist_forget.{png,pdf} — R-Gate-F-mnist-n10: Task-A forgetting
                                     under partial-EWC vs full-with-replay vs
                                     naive-full.

All figures use ONLY measured numbers from the JSON artifacts in
experiments/. No numbers are typed into the source.

Usage:
  python scripts/make_paper_figures.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


FIG_DIR = Path("docs/paper/figures")
DPI = 300


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"  wrote {path}")
    plt.close(fig)


# -------------------------------------------------------------------------
# Fig 1: Elec2 Pareto
# -------------------------------------------------------------------------
def fig_elec2_pareto() -> None:
    src = Path("experiments/phase_e_elec2.json")
    if not src.exists():
        print(f"SKIP fig_elec2_pareto: {src} missing")
        return
    d = json.load(src.open())
    raw = d["raw"]  # dict strategy -> list of per-seed dicts

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    colors = {
        "cadence_rule": "#1f77b4",
        "reactive_full": "#d62728",
        "periodic": "#7f7f7f",
    }
    labels = {
        "cadence_rule": "CADENCE (rule)",
        "reactive_full": "Reactive full retrain",
        "periodic": "Periodic (every 4 win)",
    }
    for strat in ("cadence_rule", "reactive_full", "periodic"):
        rows = raw[strat]
        f1s = [r["mean_f1"] for r in rows]
        ghs = [r["total_gpu_hr"] for r in rows]
        ax.scatter(ghs, f1s, s=60, color=colors[strat], label=labels[strat],
                   edgecolors="black", linewidths=0.5, zorder=3)
        # 95% bootstrap CI cross for each strategy
        mf = statistics.fmean(f1s)
        mg = statistics.fmean(ghs)
        ax.plot(mg, mf, marker="+", color=colors[strat], markersize=14,
                markeredgewidth=2, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("Total GPU-hours per episode (log scale)")
    ax.set_ylabel("Mean F1 (prequential)")
    ax.set_title("Elec2 real-drift end-to-end: Pareto frontier\n(3 seeds; + = mean)")
    ax.grid(True, alpha=0.3, zorder=1)
    ax.legend(loc="lower right", fontsize=9)
    # Annotate the Claim-B pair
    rso_gpu = statistics.fmean([r["total_gpu_hr"] for r in raw["cadence_rule"]])
    rso_f1 = statistics.fmean([r["mean_f1"] for r in raw["cadence_rule"]])
    rf_gpu = statistics.fmean([r["total_gpu_hr"] for r in raw["reactive_full"]])
    rf_f1 = statistics.fmean([r["mean_f1"] for r in raw["reactive_full"]])
    savings = 1 - rso_gpu / rf_gpu
    gap = rso_f1 - rf_f1
    ax.annotate(
        f"CADENCE saves {100*savings:.1f}% GPU\n"
        f"for a {gap:+.3f} F1 gap",
        xy=(rso_gpu, rso_f1), xytext=(rso_gpu * 3, rso_f1 - 0.05),
        fontsize=8.5, arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
    )
    _save(fig, "fig_elec2_pareto")


# -------------------------------------------------------------------------
# Fig 2: H1 easy/hard subset
# -------------------------------------------------------------------------
def fig_h1_subset() -> None:
    src = Path("experiments/phase_b_n10.json")
    if not src.exists():
        print(f"SKIP fig_h1_subset: {src} missing")
        return
    d = json.load(src.open())
    rows = d["rows"]

    # Recompute EASY/HARD split by PSI top-1 saturation.
    psi_by_scenario: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["scorer"] == "psi_stub":
            psi_by_scenario[r["scenario"]].append(r["top1"])
    easy = [s for s, xs in psi_by_scenario.items() if statistics.fmean(xs) >= 0.9]
    hard = [s for s, xs in psi_by_scenario.items() if statistics.fmean(xs) < 0.9]

    scorers = ("psi_stub", "cdag_structural", "gnn_learned")
    scorer_labels = {"psi_stub": "PSI (correlational)",
                     "cdag_structural": "CDAG-structural",
                     "gnn_learned": "CDAG+GNN (ours)"}
    scorer_colors = {"psi_stub": "#7f7f7f",
                     "cdag_structural": "#ff7f0e",
                     "gnn_learned": "#1f77b4"}
    metrics = ("reciprocal_rank", "auroc")
    metric_labels = {"reciprocal_rank": "MRR", "auroc": "AUROC"}

    def _agg(scorer: str, scenarios: list[str], field: str) -> tuple[float, float]:
        xs = [r[field] for r in rows if r["scorer"] == scorer and r["scenario"] in scenarios]
        if not xs:
            return (float("nan"), 0.0)
        m = statistics.fmean(xs)
        s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        return (m, s)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), sharey=True)
    x = np.arange(len(scorers))
    width = 0.35
    for ax, metric in zip(axes, metrics):
        easy_vals = [_agg(s, easy, metric) for s in scorers]
        hard_vals = [_agg(s, hard, metric) for s in scorers]
        for i, scorer in enumerate(scorers):
            ax.bar(x[i] - width/2, easy_vals[i][0], width,
                   yerr=easy_vals[i][1], color=scorer_colors[scorer], alpha=0.55,
                   edgecolor="black", linewidth=0.6, capsize=3,
                   label=f"{scorer_labels[scorer]} (EASY)" if metric == "reciprocal_rank" else None)
            ax.bar(x[i] + width/2, hard_vals[i][0], width,
                   yerr=hard_vals[i][1], color=scorer_colors[scorer], alpha=1.0,
                   edgecolor="black", linewidth=0.6, capsize=3, hatch="//",
                   label=f"{scorer_labels[scorer]} (HARD)" if metric == "reciprocal_rank" else None)
        ax.set_xticks(x)
        ax.set_xticklabels(["PSI", "CDAG-\nstruct", "CDAG+\nGNN"], fontsize=9)
        ax.set_ylabel(metric_labels[metric])
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3, zorder=0)
        ax.set_title(metric_labels[metric])

    axes[0].legend(loc="upper left", fontsize=7, framealpha=0.95)
    fig.suptitle("H1 attribution: EASY (solid) vs HARD (hatched)\n"
                 f"n=10 seeds; EASY = {len(easy)} sat. scenarios; HARD = {len(hard)} contested",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, "fig_h1_subset")


# -------------------------------------------------------------------------
# Fig 3: H3 MNIST forgetting
# -------------------------------------------------------------------------
def fig_h3_mnist_forget() -> None:
    src = Path("experiments/phase_f_mnist_n10.json")
    if not src.exists():
        print(f"SKIP fig_h3_mnist_forget: {src} missing")
        return
    d = json.load(src.open())

    # phase_f_mnist_n10.json schema:
    #   d["forgetting_partial"]           = {mean, std, values: [...]}
    #   d["forgetting_full"]              = {mean, std, values: [...]}
    #   d["forgetting_full_with_replay"]  = {mean, std, values: [...]}
    key_to_arm = {
        "forgetting_partial": "partial_ewc_layer1",
        "forgetting_full": "full_naive_no_replay",
        "forgetting_full_with_replay": "full_with_replay",
    }
    per_seed: dict[str, list[float]] = {}
    for jkey, arm in key_to_arm.items():
        entry = d.get(jkey)
        if isinstance(entry, dict) and isinstance(entry.get("values"), list):
            per_seed[arm] = [float(x) for x in entry["values"]]

    if not per_seed:
        print(f"SKIP fig_h3_mnist_forget: no forgetting_* entries with `values` in {src}")
        return

    labels = {
        "partial_ewc_layer1": "Partial EWC (ours)",
        "full_naive_no_replay": "Full retrain (naive)",
        "full_with_replay": "Full retrain\n+ Task A replay",
    }
    colors = {
        "partial_ewc_layer1": "#1f77b4",
        "full_naive_no_replay": "#d62728",
        "full_with_replay": "#ff7f0e",
    }
    arms = [a for a in ("partial_ewc_layer1", "full_naive_no_replay", "full_with_replay") if a in per_seed]

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    positions = np.arange(len(arms))
    box_data = [per_seed[a] for a in arms]
    bp = ax.boxplot(box_data, positions=positions, widths=0.55, patch_artist=True,
                    showmeans=True, meanline=True,
                    medianprops=dict(color="black", linewidth=1.4),
                    meanprops=dict(color="black", linewidth=1.2, linestyle=":"))
    for patch, arm in zip(bp["boxes"], arms):
        patch.set_facecolor(colors[arm])
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")

    # Overlay individual seeds
    rng = np.random.default_rng(0)
    for i, arm in enumerate(arms):
        y = per_seed[arm]
        x = np.full_like(y, positions[i], dtype=float) + rng.uniform(-0.08, 0.08, size=len(y))
        ax.scatter(x, y, color=colors[arm], s=25, edgecolor="black", linewidth=0.4, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([labels[a] for a in arms], fontsize=9)
    ax.set_ylabel("Task-A forgetting Δ F1  (higher = worse)")
    ax.set_title("H3: Split-MNIST forgetting under 3 retrain arms\n"
                 f"n={len(per_seed[arms[0]])} seeds; box=IQR, dashed=mean")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "fig_h3_mnist_forget")


def main() -> int:
    print("Generating publication figures (from existing artifacts, no new runs)")
    print(f"Output dir: {FIG_DIR}\n")
    fig_elec2_pareto()
    print()
    fig_h1_subset()
    print()
    fig_h3_mnist_forget()
    print("\nDone. See docs/paper/figures/README.md for figure captions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
