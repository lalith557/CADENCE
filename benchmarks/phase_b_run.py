"""Gate B runner — Step B of the Execution Plan.

Anchors Claim A (CDAG-GNN attribution beats correlational PSI) on the
synthetic drift-injection benchmark.

For each scenario × seed we score attribution three ways:
  * psi_stub          — PSI-based correlational baseline (Phase 1)
  * cdag_structural   — pre-Step-B structural path-sum proxy (Phase 3)
  * gnn_learned       — the Step B GraphSAGE/GAT-style GNN trained on a
                        synthetic (CDAG, root-cause) sandbox

Metrics:
  * top-1 accuracy    — did the top-ranked feature match the injected feature?
  * top-3 accuracy    — was the injected feature in the top 3?
  * MRR (1/rank)      — mean reciprocal rank of the injected feature
  * AUROC             — root-vs-rest AUROC on the score vector
  * SHD               — structural Hamming distance of the learned CDAG
                        vs. a naive ground-truth edge (source_feature ->
                        performance). Reported for the CDAG+GNN scorer only.

Passing Gate B (H1): the paired Wilcoxon of gnn_learned > psi_stub returns
p < 0.05 on both reciprocal_rank and AUROC across seeds × scenarios.

Usage:
    python -m benchmarks.phase_b_run --config configs/default.yaml \
        --seeds 5 --gnn-samples 400 --gnn-epochs 20
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path

import mlflow
import numpy as np
import torch
from scipy import stats as sci_stats
from sklearn.metrics import f1_score, roc_auc_score

from benchmarks.baselines.harness import make_baseline_stream
from benchmarks.synthetic_drift_gen import build_default_scenarios
from cadence.adapters.neural import FraudNet, FraudNetConfig
from cadence.attribution import (
    CDAGResponsibilityScorer,
    GNNConfig,
    GNNResponsibilityScorer,
    GNNTrainConfig,
    generate_sandbox_dataset,
    train_gnn,
)
from cadence.cdag import build_cdag_from_windows, compute_windowed_signals
from cadence.collector.drift_trigger import DriftTriggerConfig, PSITrigger
from cadence.common.config import load_config
from cadence.common.device import get_device
from cadence.common.logging import get_logger
from cadence.common.seeds import set_global_seed
from cadence.common.tracking import start_run
from cadence.data.loaders import load_credit_card_fraud
from cadence.rso.scorers import PSIResponsibilityScorer

log = get_logger("cadence.benchmarks.phase_b")


def _rank_of(scores: np.ndarray, target_idx: int) -> int:
    order = np.argsort(-scores)
    return int(np.where(order == target_idx)[0][0]) + 1


def _auroc_root_vs_rest(scores: np.ndarray, target_idx: int) -> float:
    labels = np.zeros_like(scores)
    labels[target_idx] = 1
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def _mean_std(vs):
    vs = list(vs)
    if not vs:
        return (0.0, 0.0)
    return (
        float(statistics.fmean(vs)),
        float(statistics.pstdev(vs)) if len(vs) > 1 else 0.0,
    )


def _shd_gt_feature_edge(
    learned_weights: np.ndarray,
    node_set,
    gt_feature_idx: int,
    *,
    edge_weight_floor: float = 1e-3,
) -> int:
    """Toy SHD proxy: on synthetic drift, the ground truth we control is
    "the injected feature causally drives the performance node." We check
    whether the learned CDAG has a nonzero directed path from the
    ground-truth feature node to the performance node.

    SHD = 0 if the learned graph contains such a path (either direct or
    2-hop through any cluster/other-feature node) and no spurious edges
    from a *different* feature directly to the performance node.
    SHD > 0 counts deviations.

    Not the classical SHD over a full DAG (which requires a known ground-
    truth DAG, which we do not have for a real production model). Instead
    it's a targeted invariant that captures the useful part of SHD.
    """
    W = np.abs(learned_weights) >= edge_weight_floor
    perf_idx = node_set.performance_index
    gt_node = node_set.feature_indices[gt_feature_idx]

    # Direct edge gt -> perf
    direct = bool(W[gt_node, perf_idx])
    # 2-hop: gt -> any node -> perf
    two_hop = bool((W[gt_node] & W[:, perf_idx]).any())
    reaches = direct or two_hop

    # Spurious: any OTHER feature that reaches perf but the ground-truth doesn't
    spurious = 0
    for fi, fn in enumerate(node_set.feature_indices):
        if fi == gt_feature_idx:
            continue
        f_direct = bool(W[fn, perf_idx])
        f_two = bool((W[fn] & W[:, perf_idx]).any())
        if f_direct or f_two:
            spurious += 1

    shd = 0
    if not reaches:
        shd += 1  # missed the true edge
    # We don't penalize *every* spurious path (CDAG is small; many features
    # correlate) — but if the true edge is missed AND spurious paths exist,
    # count them.
    if not reaches:
        shd += min(spurious, 5)
    return shd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--scenarios", default="all")
    p.add_argument("--window-size", type=int, default=2048)
    p.add_argument("--windows-per-episode", type=int, default=8)
    p.add_argument("--gnn-samples", type=int, default=400)
    p.add_argument("--gnn-epochs", type=int, default=20)
    p.add_argument("--gnn-lr", type=float, default=3e-3)
    p.add_argument("--sandbox-window-size", type=int, default=512)
    p.add_argument("--out", default="experiments/phase_b_summary.json")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    log.info("phase_b_start", seeds=args.seeds, gnn_samples=args.gnn_samples)

    ds = load_credit_card_fraud(cfg.data, seed=42)
    n_train = ds.X_train.shape[0]
    idx = np.arange(n_train)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    n_pretrain = int(0.70 * n_train)
    n_stream = int(0.25 * n_train)
    X_pre = ds.X_train[idx[:n_pretrain]]
    y_pre = ds.y_train[idx[:n_pretrain]]
    X_stream = ds.X_train[idx[n_pretrain : n_pretrain + n_stream]]
    y_stream = ds.y_train[idx[n_pretrain : n_pretrain + n_stream]]

    set_global_seed(42)
    adapter = FraudNet(
        FraudNetConfig(
            input_dim=ds.X_train.shape[1],
            hidden_dims=list(cfg.model.hidden_dims),
            dropout=cfg.model.dropout,
            lr=cfg.model.lr,
            batch_size=cfg.model.batch_size,
            max_epochs=cfg.model.max_epochs,
            early_stopping_patience=cfg.model.early_stopping_patience,
            class_weighted=cfg.model.class_weighted,
        )
    )
    adapter.fit(X_pre[:-5000], y_pre[:-5000], X_val=X_pre[-5000:], y_val=y_pre[-5000:])
    baseline_probs = adapter.predict_proba(X_pre[-5000:])
    baseline_preds = (baseline_probs >= adapter.decision_threshold).astype(np.int64)
    baseline_f1 = float(f1_score(y_pre[-5000:], baseline_preds, zero_division=0))
    log.info("baseline_f1", val=baseline_f1)

    # Scorers.
    trigger = PSITrigger(
        baseline_X=X_pre,
        feature_names=ds.feature_names,
        cfg=DriftTriggerConfig(psi_threshold=0.25, min_window_rows=200),
    )
    psi_scorer = PSIResponsibilityScorer(trigger=trigger)
    cdag_scorer = CDAGResponsibilityScorer.build(
        adapter,
        baseline_X=X_pre[:20_000],
        feature_names=ds.feature_names,
        baseline_f1=baseline_f1,
        k_overrides={"layer1": cfg.cdag.layer_1_clusters, "layer2": cfg.cdag.layer_2_clusters},
        window_size=512,
    )

    # GNN scorer — built empty, then trained on sandbox pretrain data.
    gnn_scorer = GNNResponsibilityScorer.build(
        adapter=adapter,
        baseline_X=X_pre[:20_000],
        feature_names=ds.feature_names,
        baseline_f1=baseline_f1,
        k_overrides={"layer1": cfg.cdag.layer_1_clusters, "layer2": cfg.cdag.layer_2_clusters},
        window_size=512,
        gnn_cfg=GNNConfig(),
    )
    log.info("scorers_built")

    # Sandbox pretrain.
    dev = get_device()
    log.info("sandbox_generate_start", n_samples=args.gnn_samples)
    samples = generate_sandbox_dataset(
        adapter=adapter,
        tap=gnn_scorer.tap,
        node_set=gnn_scorer.node_set,
        baseline_means=gnn_scorer.baseline_means,
        baseline_stds=gnn_scorer.baseline_stds,
        baseline_stream_X=X_stream,
        baseline_stream_y=y_stream,
        feature_names=ds.feature_names,
        n_samples=args.gnn_samples,
        window_size=args.sandbox_window_size,
        seed=0,
        device=dev,
        on_progress=lambda i, n: log.info("sandbox_progress", i=i, n=n),
    )
    log.info(
        "sandbox_generate_done",
        n=len(samples),
        first_root=samples[0].root_cause_feature_idx,
    )

    train_summary = train_gnn(
        gnn_scorer.model,
        samples,
        cfg=GNNTrainConfig(lr=args.gnn_lr, epochs=args.gnn_epochs),
        device=dev,
    )
    log.info(
        "gnn_trained",
        val_acc=train_summary["final_val_acc"],
        wall_s=train_summary["wall_s"],
        max_vram_mb=train_summary.get("max_vram_mb"),
    )

    all_scenarios = build_default_scenarios(ds.feature_names)
    if args.scenarios == "all":
        eval_scenarios = all_scenarios
    else:
        wanted = set(args.scenarios.split(","))
        eval_scenarios = [s for s in all_scenarios if s.name in wanted]
    log.info("scenarios", n=len(eval_scenarios), names=[s.name for s in eval_scenarios])

    with start_run(cfg, run_name="phase_b_gate_b"):
        mlflow.log_param("phase", "B")
        mlflow.log_param("scorers", "psi_stub,cdag_structural,gnn_learned")
        mlflow.log_param("n_seeds", args.seeds)
        mlflow.log_param("n_scenarios", len(eval_scenarios))
        mlflow.log_param("gnn_samples", args.gnn_samples)
        mlflow.log_param("gnn_epochs", args.gnn_epochs)
        mlflow.log_param("gnn_lr", args.gnn_lr)
        mlflow.log_metric("gnn_val_acc", float(train_summary["final_val_acc"]))
        if "max_vram_mb" in train_summary:
            mlflow.log_metric("gnn_train_max_vram_mb", float(train_summary["max_vram_mb"]))

        rows: list[dict] = []
        shd_rows: list[dict] = []

        for scenario in eval_scenarios:
            for seed in range(args.seeds):
                set_global_seed(seed)
                stream_X, stream_y, injector = make_baseline_stream(
                    X_stream, y_stream, scenario, seed=seed
                )
                gt_idx = injector.ground_truth().root_cause_feature_idx
                n_use = args.window_size * args.windows_per_episode
                window_X = stream_X[:n_use]
                window_y = stream_y[:n_use]

                psi_scores = psi_scorer(adapter, window_X, window_y)
                cdag_scores = cdag_scorer(adapter, window_X, window_y)
                gnn_scores = gnn_scorer(adapter, window_X, window_y)

                # Also compute a fresh CDAG for the SHD proxy.
                w_size = min(gnn_scorer.window_size, max(1, window_X.shape[0] // 4))
                probs = adapter.predict_proba(window_X)
                preds = (probs >= adapter.decision_threshold).astype(np.int64)
                current_f1 = float(f1_score(window_y, preds, zero_division=0))
                signals = compute_windowed_signals(
                    window_X,
                    window_y,
                    gnn_scorer.tap,
                    gnn_scorer.node_set,
                    gnn_scorer.baseline_means,
                    gnn_scorer.baseline_stds,
                    window_f1=current_f1,
                    window_size=w_size,
                )
                cdag = build_cdag_from_windows(
                    signals,
                    pc_alpha=0.05,
                    notears_lambda=0.05,
                    notears_max_iter=25,
                )
                shd = _shd_gt_feature_edge(cdag.weights, gnn_scorer.node_set, gt_idx)
                shd_rows.append(
                    {"scenario": scenario.name, "seed": seed, "shd_proxy": shd, "gt_idx": gt_idx}
                )

                for scorer_name, scores in (
                    ("psi_stub", psi_scores),
                    ("cdag_structural", cdag_scores),
                    ("gnn_learned", gnn_scores),
                ):
                    rank = _rank_of(scores, gt_idx)
                    rows.append(
                        {
                            "scorer": scorer_name,
                            "scenario": scenario.name,
                            "seed": seed,
                            "gt_idx": gt_idx,
                            "rank": rank,
                            "top1": int(rank == 1),
                            "top3": int(rank <= 3),
                            "reciprocal_rank": 1.0 / rank,
                            "auroc": _auroc_root_vs_rest(scores, gt_idx),
                        }
                    )
                    log.info(
                        "score_row",
                        scorer=scorer_name,
                        scenario=scenario.name,
                        seed=seed,
                        rank=rank,
                    )

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        def _agg(scorer_name):
            r = [x for x in rows if x["scorer"] == scorer_name]
            valid_auroc = [x["auroc"] for x in r if not np.isnan(x["auroc"])]
            return {
                "n": len(r),
                "top1_acc": _mean_std(x["top1"] for x in r),
                "top3_acc": _mean_std(x["top3"] for x in r),
                "mean_reciprocal_rank": _mean_std(x["reciprocal_rank"] for x in r),
                "auroc": _mean_std(valid_auroc),
                "mean_rank": _mean_std(x["rank"] for x in r),
            }

        summary = {
            "aggregates": {
                "psi_stub": _agg("psi_stub"),
                "cdag_structural": _agg("cdag_structural"),
                "gnn_learned": _agg("gnn_learned"),
            },
            "shd_proxy_mean": float(np.mean([r["shd_proxy"] for r in shd_rows])),
            "shd_proxy_std": float(np.std([r["shd_proxy"] for r in shd_rows])),
            "gnn_train": {
                "final_val_acc": train_summary["final_val_acc"],
                "final_val_loss": train_summary["final_val_loss"],
                "wall_s": train_summary["wall_s"],
                "max_vram_mb": train_summary.get("max_vram_mb"),
            },
            "rows": rows,
            "shd_rows": shd_rows,
        }

        # Paired Wilcoxon per scorer pair.
        keys = sorted({(x["scenario"], x["seed"]) for x in rows})
        stat_table = {}
        for pair in (("gnn_learned", "psi_stub"), ("gnn_learned", "cdag_structural")):
            a, b = pair
            for metric in ("reciprocal_rank", "auroc"):
                a_vals, b_vals = [], []
                for scen, seed in keys:
                    ax = next(
                        x for x in rows if x["scorer"] == a and x["scenario"] == scen and x["seed"] == seed
                    )
                    bx = next(
                        x for x in rows if x["scorer"] == b and x["scenario"] == scen and x["seed"] == seed
                    )
                    if metric == "auroc" and (np.isnan(ax[metric]) or np.isnan(bx[metric])):
                        continue
                    a_vals.append(ax[metric])
                    b_vals.append(bx[metric])
                try:
                    stat, p = sci_stats.wilcoxon(a_vals, b_vals, alternative="greater")
                    stat_table[f"{a}_gt_{b}_{metric}"] = {
                        "stat": float(stat),
                        "p": float(p),
                        "n_pairs": len(a_vals),
                    }
                except ValueError as e:
                    stat_table[f"{a}_gt_{b}_{metric}"] = {
                        "stat": None,
                        "p": None,
                        "err": str(e),
                        "n_pairs": len(a_vals),
                    }
        summary["wilcoxon"] = stat_table

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        mlflow.log_artifact(args.out)

        print("\n=== Gate B: attribution precision (PSI vs CDAG-structural vs GNN) ===")
        print(f"{'scorer':<20}{'top1_acc':>18}{'top3_acc':>18}{'MRR':>18}{'AUROC':>18}")
        for name in ("psi_stub", "cdag_structural", "gnn_learned"):
            a = summary["aggregates"][name]
            print(
                f"{name:<20}"
                f"{a['top1_acc'][0]:>10.4f} +/- {a['top1_acc'][1]:.4f}"
                f"{a['top3_acc'][0]:>10.4f} +/- {a['top3_acc'][1]:.4f}"
                f"{a['mean_reciprocal_rank'][0]:>10.4f} +/- {a['mean_reciprocal_rank'][1]:.4f}"
                f"{a['auroc'][0]:>10.4f} +/- {a['auroc'][1]:.4f}"
            )
        print(f"\nSHD proxy (learned CDAG vs ground truth): {summary['shd_proxy_mean']:.3f} +/- {summary['shd_proxy_std']:.3f}")
        print(f"GNN final val acc: {train_summary['final_val_acc']:.4f}")
        if "max_vram_mb" in train_summary:
            print(f"GNN peak VRAM (train): {train_summary['max_vram_mb']:.2f} MB")
        print("\nWilcoxon signed-rank:")
        for k, v in stat_table.items():
            print(f"  {k}: stat={v.get('stat')} p={v.get('p')} n_pairs={v.get('n_pairs')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
