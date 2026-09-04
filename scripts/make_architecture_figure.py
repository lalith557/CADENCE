#!/usr/bin/env python
"""CADENCE system architecture figure for the paper.

Renders the closed-loop pipeline as a clean publication schematic (PNG+PDF).
Pure matplotlib, no external assets. Boxes + arrows only; theme-neutral.

Output: docs/paper/figures/fig_architecture.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_DIR = Path("docs/paper/figures")


def _box(ax, x, y, w, h, title, sub, face):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.3, edgecolor="#333333", facecolor=face, zorder=2))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=9.4, fontweight="bold", zorder=3)
    if sub:
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=7.1, color="#333333", zorder=3, style="italic")


def _arrow(ax, p0, p1, color="#222222", style="-|>", rad=0.0, lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=13, lw=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=1, linestyle=ls))


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Palette (colour-blind-safe, muted).
    c_data = "#dbe9f6"     # blue-ish  (telemetry / model)
    c_attr = "#e6f0da"     # green-ish (attribution)
    c_dec = "#fbe6cf"      # orange-ish (decision)
    c_exec = "#f6dede"     # red-ish   (execution)

    # Row 1 (top): live loop
    _box(ax, 0.2, 4.7, 1.9, 0.95, "Production Model", "score + tap activations\n(neural / tree / text)", c_data)
    _box(ax, 2.5, 4.7, 1.9, 0.95, "Streaming Collector", "PII-safe feature +\nactivation windows", c_data)
    _box(ax, 4.8, 4.7, 2.0, 0.95, "Drift Trigger", "PSI + delayed-label F1\n(the smoke alarm)", c_data)

    # Row 2 (middle): attribution
    _box(ax, 4.8, 3.1, 2.0, 0.95, "CDAG Builder", "PC + NOTEARS over\nfeature & activation nodes", c_attr)
    _box(ax, 2.5, 3.1, 1.9, 0.95, "GNN + Surrogate", "responsibility score\nper node (no ground-truth\nintervention)", c_attr)

    # Row 3 (bottom): decision + execution
    _box(ax, 2.5, 1.5, 1.9, 0.95, "RSO (PPO)", "no-op / partial / full\ncost+carbon+SLA reward", c_dec)
    _box(ax, 4.8, 1.5, 2.0, 0.95, "Executor", "EWC partial / full retrain\n+ shadow-canary validate", c_exec)
    _box(ax, 7.1, 1.5, 2.0, 0.95, "Promote / Rollback", "shadow F1 >= SLA?\nrollback -> surrogate", c_exec)

    # Supporting service
    _box(ax, 7.1, 3.1, 2.0, 0.95, "Cost / Carbon Model", "FLOPs x grid intensity\n(CodeCarbon)", c_attr)

    # Arrows — top row L->R
    _arrow(ax, (2.1, 5.17), (2.5, 5.17))
    _arrow(ax, (4.4, 5.17), (4.8, 5.17))
    # trigger -> CDAG (down)
    _arrow(ax, (5.8, 4.7), (5.8, 4.05))
    # CDAG -> GNN (left)
    _arrow(ax, (4.8, 3.57), (4.4, 3.57))
    # cost model -> RSO (feeds reward): route along
    _arrow(ax, (7.1, 3.57), (3.45, 3.57 - 0.001), color="#888888", rad=-0.12, ls=(0, (4, 3)))
    ax.text(5.6, 3.9, "cost/carbon estimate", fontsize=6.6, color="#777777", ha="center")
    # GNN -> RSO (down)
    _arrow(ax, (3.45, 3.1), (3.45, 2.45))
    ax.text(3.7, 2.78, "scores", fontsize=6.8, color="#555555", ha="left")
    # RSO -> Executor
    _arrow(ax, (4.4, 1.97), (4.8, 1.97))
    # Executor -> Promote
    _arrow(ax, (6.8, 1.97), (7.1, 1.97))
    # Promote -> back to Production Model (big feedback arc, left+up)
    _arrow(ax, (8.1, 2.45), (1.15, 4.7), color="#1f6f3f", rad=0.28, lw=1.7)
    ax.text(3.0, 0.72, "promoted model redeploys  →  loop resumes monitoring",
            fontsize=7.6, color="#1f6f3f", ha="center", style="italic")

    # Phase brackets / labels on the left
    ax.text(0.05, 5.9, "CADENCE closed loop", fontsize=11.5, fontweight="bold")
    for y, lab, col in [(5.17, "MONITOR", "#2b6cb0"),
                        (3.35, "ATTRIBUTE (why)", "#357a38"),
                        (1.75, "DECIDE + REPAIR (how much)", "#b5651d")]:
        ax.text(0.05, y, lab, fontsize=7.4, fontweight="bold", color=col,
                rotation=90, va="center", ha="center")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = FIG_DIR / f"fig_architecture.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
