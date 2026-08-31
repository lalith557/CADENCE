"""Generate paper-ready PNGs from experiments/*.json artifacts.

One figure per R-entry that has plot-worthy data. Output goes to
`docs/paper/figures/`. Every function is guarded — a missing artifact
skips that figure with a printed warning rather than crashing the whole
script, so partial results still produce partial figures.

Usage:
    python -m scripts.build_paper_figures
    python -m scripts.build_paper_figures --dpi 200 --out docs/paper/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no interactive backend needed
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"


def _load(name: str) -> dict | None:
    p = EXP / name
    if not p.exists():
        print(f"[skip] {name} not found")
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save(fig: plt.Figure, out_dir: Path, name: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok]   wrote {path.relative_to(ROOT)}")


# ---------- H1 attribution: scorer comparison bar chart ----------


def fig_h1_scorer_comparison(out_dir: Path, dpi: int) -> None:
    summary = _load("phase_b_n10.json")
    if summary is None:
        return
    agg = summary.get("aggregates", {})
    if not agg:
        return
    scorers = ["psi_stub", "cdag_structural", "gnn_learned"]
    metrics = ["top1_acc", "top3_acc", "mean_reciprocal_rank", "auroc"]
    labels = ["top-1 acc", "top-3 acc", "MRR", "AUROC"]
    colours = {"psi_stub": "#a4c8ff", "cdag_structural": "#f6c85f", "gnn_learned": "#00c9c9"}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(metrics))
    width = 0.26
    for i, s in enumerate(scorers):
        means = []
        stds = []
        for m in metrics:
            v = agg.get(s, {}).get(m, [0.0, 0.0])
            means.append(float(v[0]))
            stds.append(float(v[1]))
        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=s,
            color=colours[s],
            edgecolor="#333",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (mean ± std across 50 paired samples)")
    ax.set_title("H1 attribution — 5 scenarios × 10 seeds (R-Gate-B-n10)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, "h1_scorer_comparison.png", dpi)


# ---------- H3 Split-MNIST forgetting ----------


def fig_h3_split_mnist(out_dir: Path, dpi: int) -> None:
    summary = _load("phase_f_mnist_n10.json")
    if summary is None:
        return
    paths = [
        ("partial\nEWC λ=5000", summary.get("forgetting_partial", {})),
        ("full\n+ replay", summary.get("forgetting_full_with_replay", {})),
        ("full\nnaive", summary.get("forgetting_full", {})),
    ]
    means = [float(p[1].get("mean", 0.0)) for p in paths]
    stds = [float(p[1].get("std", 0.0)) for p in paths]
    labels = [p[0] for p in paths]
    colours = ["#00c9c9", "#f6c85f", "#e64c4c"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(paths))
    ax.bar(x, means, yerr=stds, capsize=5, color=colours, edgecolor="#333", linewidth=0.5)
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Task-A forgetting  (pre_F1 − post_F1)")
    ax.set_title(
        "H3 on Split-MNIST {0,1} → {2,3}, 10 seeds  (R-Gate-F-mnist-n10)\n"
        "Wilcoxon partial<naive p=0.001;  partial<replay p=0.001"
    )
    for i, (m, s) in enumerate(zip(means, stds, strict=True)):
        ax.text(i, m + (0.02 if m >= 0 else -0.04), f"{m:+.3f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, "h3_split_mnist_forgetting.png", dpi)


# ---------- Elec2 real-drift trajectory ----------


def fig_elec2_f1_over_time(out_dir: Path, dpi: int) -> None:
    summary = _load("phase_e_elec2.json")
    if summary is None:
        return
    raw = summary.get("raw", {})
    if not raw:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colours = {"cadence_rule": "#00c9c9", "periodic": "#f6c85f", "reactive_full": "#e64c4c"}
    for strat, seeds in raw.items():
        # Stack per-seed rolling F1 and average.
        curves = [r.get("rolling_f1", []) for r in seeds]
        max_len = max(len(c) for c in curves) if curves else 0
        if max_len == 0:
            continue
        padded = np.array([c + [np.nan] * (max_len - len(c)) for c in curves])
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, marker="o", label=strat, color=colours.get(strat, "grey"))
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=colours.get(strat, "grey"))
    ax.axhline(summary.get("config", {}).get("sla", 0.65), linestyle="--", color="#666", label="SLA")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Mean F1 (± std across seeds)")
    ax.set_title("Elec2 real drift — F1 trajectory per strategy  (R-Gate-E)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "elec2_f1_over_time.png", dpi)


# ---------- LightGBM tree Pareto ----------


def fig_tree_pareto(out_dir: Path, dpi: int) -> None:
    summary = _load("phase_f_tree_sla045.json")
    if summary is None:
        return
    agg = summary.get("aggregates", {})
    if not agg:
        return

    def _extract(strat: str) -> tuple[float, float]:
        v = agg.get(strat, {})
        f1 = v.get("mean_f1", [0.0, 0.0])[0]
        # Actions total (proxy for cost — every action here is a full retrain).
        full = v.get("actions_full", 0.0)
        return float(f1), float(full)

    strategies = ["cadence_rule", "periodic", "reactive_full"]
    colours = ["#00c9c9", "#f6c85f", "#e64c4c"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for s, c in zip(strategies, colours, strict=True):
        f1, cost = _extract(s)
        ax.scatter(cost, f1, s=180, color=c, edgecolor="#333", zorder=5, label=s)
        ax.annotate(
            s, xy=(cost, f1), xytext=(6, 6), textcoords="offset points", fontsize=9,
        )
    ax.set_xlabel("Full-retrain actions (proxy for compute cost)")
    ax.set_ylabel("Mean F1")
    ax.set_title(
        "LightGBM tree adapter on GMSC at SLA=0.45\n"
        "CADENCE = Pareto-frontier position  (R-Gate-F-tree-sla045)"
    )
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "tree_pareto_frontier.png", dpi)


# ---------- Robustness — GNN AUROC vs edge dropout ----------


def fig_robustness_dropout(out_dir: Path, dpi: int) -> None:
    summary = _load("phase_f_robustness.json")
    if summary is None:
        return
    ntl = summary.get("noisy_telemetry_auroc", {})
    if not ntl:
        return
    fracs = sorted(float(k) for k in ntl.keys())
    means = [float(ntl[str(f)][0]) for f in fracs]
    stds = [float(ntl[str(f)][1]) for f in fracs]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(fracs, means, yerr=stds, marker="o", capsize=5, color="#00c9c9", linewidth=2)
    ax.set_xlabel("Fraction of CDAG edges randomly dropped")
    ax.set_ylabel("GNN attribution AUROC")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "GNN robustness under noisy telemetry  (R-Gate-F)\n"
        "Half the CDAG edges dropped → only ~4pp AUROC loss"
    )
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "robustness_edge_dropout.png", dpi)


# ---------- False-alarm vs true-alert rate ----------


def fig_false_alarm(out_dir: Path, dpi: int) -> None:
    summary = _load("phase_f_robustness.json")
    if summary is None:
        return
    fa = summary.get("false_alarm_rate_undrifted", {}).get("mean")
    tr = summary.get("true_alert_rate_drifted", {}).get("mean")
    if fa is None or tr is None:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(
        ["undrifted stream\n(false-alarm)", "drifted stream\n(true-alert)"],
        [float(fa), float(tr)],
        color=["#a4c8ff", "#e64c4c"],
        edgecolor="#333",
    )
    for b, v in zip(bars, [float(fa), float(tr)], strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fraction of windows that fired the trigger")
    ax.set_title("PSI trigger sensitivity vs specificity  (R-Gate-F)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, "false_alarm_vs_true_alert.png", dpi)


# ---------- Surrogate calibration ----------


def fig_surrogate_calibration(out_dir: Path, dpi: int) -> None:
    summary = _load("phase_c_smoke.json") or _load("phase_c_smoke2.json")
    if summary is None:
        return
    jt = summary.get("joint_train", {})
    preds = jt.get("val_preds")
    targets = jt.get("val_targets")
    if not preds or not targets:
        return
    preds = np.asarray(preds, dtype=float)
    targets = np.asarray(targets, dtype=float)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(targets, preds, s=40, color="#00c9c9", edgecolor="#333", alpha=0.85)
    ax.plot([0, 1], [0, 1], "--", color="#666", label="perfect calibration")
    r2 = jt.get("final_val_r2", "n/a")
    mae = jt.get("final_val_mae", "n/a")
    ax.set_xlabel("Measured post-fix F1")
    ax.set_ylabel("Surrogate-predicted post-fix F1")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    r2_str = f"{r2:.3f}" if isinstance(r2, (int, float)) else str(r2)
    mae_str = f"{mae:.3f}" if isinstance(mae, (int, float)) else str(mae)
    ax.set_title(f"Surrogate calibration on held-out sandbox val split\n" f"R²={r2_str}  MAE={mae_str}  (R-Gate-C)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "surrogate_calibration.png", dpi)


# ---------- Main ----------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/paper/figures")
    p.add_argument("--dpi", type=int, default=180)
    args = p.parse_args()

    out_dir = ROOT / args.out
    fig_h1_scorer_comparison(out_dir, args.dpi)
    fig_h3_split_mnist(out_dir, args.dpi)
    fig_elec2_f1_over_time(out_dir, args.dpi)
    fig_tree_pareto(out_dir, args.dpi)
    fig_robustness_dropout(out_dir, args.dpi)
    fig_false_alarm(out_dir, args.dpi)
    fig_surrogate_calibration(out_dir, args.dpi)
    print(f"\nfigures under {out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
