"""CADENCE dashboard — the "why it broke" view.

Streamlit app that reads the experiments/phase_*_summary.json artifacts
produced by the benchmark runners and renders four panels:

  1. Live drift score — mean F1 over time per strategy.
  2. CDAG graph — nodes + weighted edges from the last built CDAG in the
     selected summary. Rendered via plotly graph_objects (no external
     graphviz install needed on Windows).
  3. Responsibility scores — per-feature bar chart from the selected
     summary's attribution rows.
  4. Decision + cost — the executor's chosen action per window + running
     cost/carbon totals, and cost saved vs the reactive_full baseline.

The dashboard deliberately does NOT retrain models — it's a read-only
view of already-committed experiment artifacts. This means it is safe
to hand to a non-engineering stakeholder and always deterministic.

Run locally:
    streamlit run dashboard/app.py
Or via docker-compose:
    docker-compose up dashboard
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"


st.set_page_config(page_title="CADENCE — why it broke", layout="wide")
st.title("CADENCE — why it broke, and what to do about it")
st.caption(
    "Causal Attribution-Driven Efficient Continual Retraining. This view "
    "differentiates CADENCE from Evidently/WhyLabs: those tell you *that* "
    "your model drifted; this shows *why*, at what internal layer, and "
    "which minimal-scope retrain would fix it."
)


# ---------- helpers ----------


def _list_summaries() -> list[Path]:
    """Every *_summary.json / *_smoke*.json artifact in experiments/."""
    if not EXPERIMENTS_DIR.exists():
        return []
    return sorted(EXPERIMENTS_DIR.glob("phase_*.json"))


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_field(d: dict, *paths) -> object | None:
    for p in paths:
        cur = d
        try:
            for key in p:
                cur = cur[key]
            return cur
        except (KeyError, TypeError):
            continue
    return None


def _f1_series_panel(summary: dict) -> None:
    st.subheader("Live drift & recovery — mean F1 over episode")

    # phase_e_run stores results[strategy] = list of dicts with rolling_f1.
    rows: list[dict] = []
    raw = _pick_field(summary, ("raw",))
    if raw:
        for strat, seed_runs in raw.items():
            for i, r in enumerate(seed_runs):
                for step, f1 in enumerate(r.get("rolling_f1") or []):
                    rows.append({"strategy": strat, "seed": i, "step": step, "F1": f1})

    if not rows:
        # Fall back to phase_a-style single-strategy layout.
        st.info("No per-window rolling F1 in this summary; showing aggregates instead.")
        agg = _pick_field(summary, ("aggregates",))
        if agg:
            frame = pd.DataFrame(
                [
                    {
                        "strategy": s,
                        "mean_F1": v["mean_f1"][0] if isinstance(v.get("mean_f1"), list) else None,
                    }
                    for s, v in agg.items()
                ]
            )
            st.dataframe(frame, hide_index=True, use_container_width=True)
        return

    df = pd.DataFrame(rows)
    fig = go.Figure()
    for strat in df["strategy"].unique():
        d = df[df["strategy"] == strat].groupby("step")["F1"].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=d["step"], y=d["F1"], mode="lines+markers", name=strat)
        )
    fig.update_layout(
        xaxis_title="Window index",
        yaxis_title="Mean F1 (across seeds)",
        height=380,
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _cdag_panel(summary: dict) -> None:
    st.subheader("CDAG — causal graph over features + activation clusters")

    # NEW: prefer the raw NOTEARS edges from shd_rows (Step W-30 fix). If
    # the summary lacks them (older artifacts), fall back to the illustrative
    # layout so old smoke runs still open.
    shd_rows = _pick_field(summary, ("shd_rows",)) or []
    real_rows = [r for r in shd_rows if r.get("cdag_edges")]
    if real_rows:
        scenarios = sorted({r["scenario"] for r in real_rows})
        sel_scen = st.selectbox("Scenario", scenarios, key="cdag_scen")
        sel_rows = [r for r in real_rows if r["scenario"] == sel_scen]
        seeds = sorted({r["seed"] for r in sel_rows})
        sel_seed = st.selectbox("Seed", seeds, key="cdag_seed")
        row = next(r for r in sel_rows if r["seed"] == sel_seed)
        _render_real_cdag(row)
        return

    # Fallback: illustrative layout (W-30 legacy path).
    dataset = _pick_field(summary, ("dataset",))
    scenario = _pick_field(summary, ("scenario", "config", "eval_scenario"))
    n_feats = 30
    if isinstance(dataset, dict) and "n_features" in dataset:
        n_feats = int(dataset["n_features"])
    feat_names = [f"f{i}" for i in range(min(n_feats, 30))]
    labels = [*feat_names, "L1_c0", "L1_c1", "L2_c0", "performance"]
    xs = np.array(
        [np.cos(2 * np.pi * i / len(feat_names)) * 1.4 for i in range(len(feat_names))]
        + [0.6, 0.9, 0.3, 0.0],
        dtype=float,
    )
    ys = np.array(
        [np.sin(2 * np.pi * i / len(feat_names)) * 1.4 for i in range(len(feat_names))]
        + [-0.4, 0.4, 0.8, 0.0],
        dtype=float,
    )
    node_trace = go.Scatter(
        x=xs, y=ys, mode="markers+text", text=labels, textposition="top center",
        marker=dict(
            size=[16] * len(feat_names) + [22, 22, 22, 32],
            color=["#5aa6ff"] * len(feat_names) + ["#f6c85f", "#f6c85f", "#f6c85f", "#e64c4c"],
            line=dict(width=1, color="#2d2d2d"),
        ),
        hoverinfo="text",
    )
    fig = go.Figure(data=[node_trace])
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=440,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "**Illustrative CDAG layout** — this artifact predates the W-30 fix "
        "and doesn't ship the raw NOTEARS weights. Rerun `python -m benchmarks.phase_b_run "
        "--out experiments/<name>.json` and reload."
        + (f"  Scenario: {scenario}" if scenario else "")
    )


def _render_real_cdag(row: dict) -> None:
    """Render the actual NOTEARS-weighted DAG from a Phase-B shd_row entry."""
    labels: list[str] = list(row.get("node_labels", []))
    edges = row.get("cdag_edges", [])
    perf_idx = int(row.get("performance_index", len(labels) - 1))
    feat_idx = set(row.get("feature_indices", []))
    gt = int(row.get("gt_idx", -1))
    n = len(labels)
    xs = np.zeros(n, dtype=float)
    ys = np.zeros(n, dtype=float)
    # Layout: features on the left ring, clusters in the middle, performance on
    # the right.
    ring_feats = [i for i in range(n) if i in feat_idx]
    clusters = [i for i in range(n) if i not in feat_idx and i != perf_idx]
    for k, i in enumerate(ring_feats):
        angle = 2 * np.pi * k / max(len(ring_feats), 1)
        xs[i] = -1.6 + 0.3 * np.cos(angle)
        ys[i] = np.sin(angle) * 1.2
    for k, i in enumerate(clusters):
        xs[i] = 0.4 + 0.3 * ((k % 3) - 1)
        ys[i] = 1.0 - 0.35 * (k // 3)
    xs[perf_idx] = 1.8
    ys[perf_idx] = 0.0
    # Node colours: red = performance, gold = clusters, blue = features, cyan =
    # ground-truth root cause feature (bigger + brighter).
    colours = []
    sizes = []
    for i in range(n):
        if i == perf_idx:
            colours.append("#e64c4c"); sizes.append(28)
        elif i == gt:
            colours.append("#00c9c9"); sizes.append(26)
        elif i in feat_idx:
            colours.append("#5aa6ff"); sizes.append(14)
        else:
            colours.append("#f6c85f"); sizes.append(20)
    node_trace = go.Scatter(
        x=xs, y=ys, mode="markers+text", text=labels, textposition="top center",
        marker=dict(size=sizes, color=colours, line=dict(width=1, color="#2d2d2d")),
        hoverinfo="text",
    )
    edge_traces = []
    if edges:
        max_w = max(abs(float(e["w"])) for e in edges) or 1.0
        for e in edges:
            s, d = int(e["src"]), int(e["dst"])
            w = abs(float(e["w"])) / max_w
            edge_traces.append(
                go.Scatter(
                    x=[xs[s], xs[d], None],
                    y=[ys[s], ys[d], None],
                    mode="lines",
                    line=dict(width=0.5 + w * 4, color="rgba(80,80,80,0.55)"),
                    hoverinfo="none",
                )
            )
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=440,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"**Real CDAG.** {len(edges)} edges above 1e-3 · scenario "
        f"`{row['scenario']}` · seed {row['seed']} · ground-truth root cause = "
        f"`{labels[gt]}` (index {gt}) · SHD proxy = {row.get('shd_proxy', 'n/a')}"
    )


def _responsibility_panel(summary: dict) -> None:
    st.subheader("Responsibility scores — which feature is driving the drop")

    # Prefer the raw per-feature score vectors from shd_rows (W-30 fix).
    shd_rows = _pick_field(summary, ("shd_rows",)) or []
    scored_rows = [r for r in shd_rows if r.get("per_scorer_feature_scores")]
    if scored_rows:
        scenarios = sorted({r["scenario"] for r in scored_rows})
        sel_scen = st.selectbox("Scenario", scenarios, key="resp_scen")
        seeds = sorted({r["seed"] for r in scored_rows if r["scenario"] == sel_scen})
        sel_seed = st.selectbox("Seed", seeds, key="resp_seed")
        row = next(
            r for r in scored_rows if r["scenario"] == sel_scen and r["seed"] == sel_seed
        )
        gt = int(row.get("gt_idx", -1))
        labels = list(row.get("node_labels", []))
        feat_indices = list(row.get("feature_indices", []))
        feat_names = [labels[i] for i in feat_indices] if labels else [f"f{i}" for i in feat_indices]
        for scorer_name, scores in row["per_scorer_feature_scores"].items():
            df = pd.DataFrame({"feature": feat_names, "score": scores})
            df = df.sort_values("score", ascending=False).head(10)
            colour = ["#00c9c9" if f == labels[gt] else "#5aa6ff" for f in df["feature"]]
            fig = go.Figure(
                data=[go.Bar(x=df["feature"], y=df["score"], marker_color=colour)]
            )
            fig.update_layout(
                title=f"{scorer_name} — top 10 (ground truth `{labels[gt]}` in teal)",
                height=260,
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        return

    # Fallback: older artifacts with a `rows` list of ranking metrics.
    rows = _pick_field(summary, ("rows",))
    if not rows:
        st.info(
            "This summary has no attribution rows. Load a `phase_b_smoke*.json` "
            "or `phase_c_smoke*.json` to see per-feature responsibility."
        )
        return
    df = pd.DataFrame(rows)
    if "scorer" in df.columns and "gt_idx" in df.columns:
        pivot = (
            df.groupby(["scorer", "scenario"])
            .agg(mean_rank=("rank", "mean"), auroc=("auroc", "mean"))
            .reset_index()
        )
        st.dataframe(pivot, hide_index=True, use_container_width=True)
    else:
        st.write(df.head())


def _decision_panel(summary: dict) -> None:
    st.subheader("Decision + cost — action per window, cost saved vs baselines")
    agg = _pick_field(summary, ("aggregates",))
    if not agg:
        st.info("No per-strategy aggregates in this summary.")
        return

    rows: list[dict] = []
    for strat, v in agg.items():
        row = {"strategy": strat}
        for k in ("mean_f1", "min_f1", "gpu_hr", "kg_co2"):
            if isinstance(v.get(k), list):
                row[f"{k}_mean"] = round(v[k][0], 5)
                row[f"{k}_std"] = round(v[k][1], 5)
            elif isinstance(v.get(k), dict):
                row[f"{k}_mean"] = round(v[k].get("mean", 0), 5)
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Cost-saved bar.
    if "cadence_rule" in agg and "reactive_full" in agg:
        cad_cost = agg["cadence_rule"].get("gpu_hr", [0, 0])
        base_cost = agg["reactive_full"].get("gpu_hr", [0, 0])
        cad_mean = cad_cost[0] if isinstance(cad_cost, list) else cad_cost.get("mean", 0)
        base_mean = base_cost[0] if isinstance(base_cost, list) else base_cost.get("mean", 0)
        if base_mean > 0:
            saved = max(0.0, 1.0 - cad_mean / base_mean)
            st.metric(
                "GPU-hr saved vs reactive_full baseline",
                f"{saved * 100:.1f}%",
                delta=f"cadence {cad_mean:.6f}  vs  baseline {base_mean:.6f}",
            )

    verdict = _pick_field(summary, ("gate_e_verdict",))
    if isinstance(verdict, dict):
        st.markdown("**Gate E verdict per baseline:**")
        for base_name, v in verdict.items():
            wins = "PASS" if v.get("cadence_wins_cost") else "FAIL (Pareto option, not strict dom.)"
            st.write(
                f"- vs `{base_name}`: F1 gap {v.get('cadence_mean_f1_gap', 0):+.4f}, "
                f"GPU-hr ratio {v.get('cadence_gpu_hr_ratio', 0):.3f} → **{wins}**"
            )


# ---------- main ----------

summaries = _list_summaries()
if not summaries:
    st.error(
        "No experiment artifacts found under `experiments/`. Run a phase "
        "runner first (e.g. `python -m benchmarks.phase_e_run --dataset elec2`)."
    )
    st.stop()

with st.sidebar:
    st.markdown("### Artifact")
    choices = {p.name: p for p in summaries}
    picked = st.selectbox("Summary JSON to visualize", list(choices.keys()))
    summary_path = choices[picked]
    st.caption(str(summary_path.relative_to(ROOT)))
    st.markdown("---")
    st.markdown("**Panels rendered:**")
    st.markdown("1. F1 over time per strategy")
    st.markdown("2. CDAG structure")
    st.markdown("3. Responsibility scores")
    st.markdown("4. Decision + cost saved")

summary = _load(summary_path)
_f1_series_panel(summary)
col_a, col_b = st.columns([3, 2])
with col_a:
    _cdag_panel(summary)
with col_b:
    _responsibility_panel(summary)
_decision_panel(summary)

st.markdown("---")
st.caption(
    "Reproducibility: every commit in `git log` includes the exact CLI to "
    "regenerate the artifact. This dashboard never trains or mutates state."
)
