"""Phase 3 gate runner — H1: CDAG attribution precision > PSI baseline.

For each drift scenario (known ground-truth root cause), have both scorers
rank the features by responsibility. Score them on:
  - top-1 accuracy (did the top pick match the true root cause?)
  - reciprocal rank (1 / rank_of_true_root)
  - AUROC (ranking true root against non-root features)

Aggregate over seeds and scenarios, run paired Wilcoxon on each metric.

Usage:
    python -m benchmarks.phase3_run --config configs/default.yaml --seeds 10 --window-size 2048
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
from sklearn.metrics import roc_auc_score

from benchmarks.baselines.harness import make_baseline_stream
from benchmarks.synthetic_drift_gen import build_default_scenarios
from cadence.adapters.neural import FraudNet, FraudNetConfig
from cadence.attribution import CDAGResponsibilityScorer
from cadence.collector.drift_trigger import DriftTriggerConfig, PSITrigger
from cadence.common.config import load_config
from cadence.common.logging import get_logger
from cadence.common.seeds import set_global_seed
from cadence.common.tracking import start_run
from cadence.data.loaders import load_credit_card_fraud
from cadence.rso.scorers import PSIResponsibilityScorer

log = get_logger("cadence.benchmarks.phase3")


def _rank_of(scores: np.ndarray, target_idx: int) -> int:
    """Return the 1-indexed rank of `target_idx` in the descending sort of `scores`."""
    order = np.argsort(-scores)
    rank = int(np.where(order == target_idx)[0][0]) + 1
    return rank


def _auroc_root_vs_rest(scores: np.ndarray, target_idx: int) -> float:
    """Root cause is the positive class (1), everything else is negative (0)."""
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
    return (float(statistics.fmean(vs)), float(statistics.pstdev(vs)) if len(vs) > 1 else 0.0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--scenarios", default="all")
    p.add_argument("--window-size", type=int, default=2048)
    p.add_argument("--windows-per-episode", type=int, default=8)
    p.add_argument("--out", default="experiments/phase3_summary.json")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    log.info("phase3_start", seeds=args.seeds)

    # Pretrained FraudNet — same seed=42 pretrain as Phase 1/2 so the baseline
    # model is identical across gates.
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
    from sklearn.metrics import f1_score

    baseline_probs = adapter.predict_proba(X_pre[-5000:])
    baseline_preds = (baseline_probs >= adapter.decision_threshold).astype(np.int64)
    baseline_f1 = float(f1_score(y_pre[-5000:], baseline_preds, zero_division=0))
    log.info("baseline_f1", val=baseline_f1)

    # Build both scorers (built ONCE on baseline data, reused across scenarios).
    trigger = PSITrigger(
        baseline_X=X_pre,
        feature_names=ds.feature_names,
        cfg=DriftTriggerConfig(psi_threshold=0.25, min_window_rows=200),
    )
    psi_scorer = PSIResponsibilityScorer(trigger=trigger)
    cdag_scorer = CDAGResponsibilityScorer.build(
        adapter,
        baseline_X=X_pre[:20_000],  # cap for k-means fit speed
        feature_names=ds.feature_names,
        baseline_f1=baseline_f1,
        k_overrides={"layer1": cfg.cdag.layer_1_clusters, "layer2": cfg.cdag.layer_2_clusters},
        window_size=512,
    )

    # Scenarios.
    all_scenarios = build_default_scenarios(ds.feature_names)
    if args.scenarios == "all":
        eval_scenarios = all_scenarios
    else:
        wanted = set(args.scenarios.split(","))
        eval_scenarios = [s for s in all_scenarios if s.name in wanted]
    log.info("scenarios", n=len(eval_scenarios), names=[s.name for s in eval_scenarios])

    with start_run(cfg, run_name="phase3-h1"):
        mlflow.log_param("phase", 3)
        mlflow.log_param("scorers", "psi_stub,cdag_structural")
        mlflow.log_param("n_seeds", args.seeds)
        mlflow.log_param("n_scenarios", len(eval_scenarios))

        # Per (scorer, scenario, seed): metrics
        rows: list[dict] = []

        for scenario in eval_scenarios:
            for seed in range(args.seeds):
                set_global_seed(seed)
                stream_X, stream_y, injector = make_baseline_stream(
                    X_stream, y_stream, scenario, seed=seed
                )
                gt_idx = injector.ground_truth().root_cause_feature_idx

                # Grab the first `windows_per_episode` × window_size rows for scoring.
                n_use = args.window_size * args.windows_per_episode
                window_X = stream_X[:n_use]
                window_y = stream_y[:n_use]

                # Score with each scorer on the SAME window.
                psi_scores = psi_scorer(adapter, window_X, window_y)
                cdag_scores = cdag_scorer(adapter, window_X, window_y)

                for scorer_name, scores in (
                    ("psi_stub", psi_scores),
                    ("cdag_structural", cdag_scores),
                ):
                    rank = _rank_of(scores, gt_idx)
                    top1 = int(rank == 1)
                    top3 = int(rank <= 3)
                    rr = 1.0 / rank
                    auroc = _auroc_root_vs_rest(scores, gt_idx)
                    rows.append(
                        {
                            "scorer": scorer_name,
                            "scenario": scenario.name,
                            "seed": seed,
                            "gt_idx": gt_idx,
                            "rank": rank,
                            "top1": top1,
                            "top3": top3,
                            "reciprocal_rank": rr,
                            "auroc": auroc,
                        }
                    )
                    log.info(
                        "score_row",
                        scorer=scorer_name,
                        scenario=scenario.name,
                        seed=seed,
                        rank=rank,
                        auroc=auroc,
                    )
                # Free memory between seeds.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Aggregate.
        def _agg(scorer_name):
            r = [x for x in rows if x["scorer"] == scorer_name]
            return {
                "n": len(r),
                "top1_acc": _mean_std(x["top1"] for x in r),
                "top3_acc": _mean_std(x["top3"] for x in r),
                "mean_reciprocal_rank": _mean_std(x["reciprocal_rank"] for x in r),
                "auroc": _mean_std(x["auroc"] for x in r if not np.isnan(x["auroc"])),
                "mean_rank": _mean_std(x["rank"] for x in r),
            }

        summary = {
            "aggregates": {
                "psi_stub": _agg("psi_stub"),
                "cdag_structural": _agg("cdag_structural"),
            },
            "rows": rows,
        }

        # Paired Wilcoxon per scenario, on reciprocal_rank and AUROC.
        # Pair by (scenario, seed).
        stat_table = {}
        for metric in ("reciprocal_rank", "auroc"):
            keys = sorted({(x["scenario"], x["seed"]) for x in rows})
            psi_vals, cdag_vals = [], []
            for scen, seed in keys:
                a = next(
                    x
                    for x in rows
                    if x["scorer"] == "psi_stub" and x["scenario"] == scen and x["seed"] == seed
                )
                b = next(
                    x
                    for x in rows
                    if x["scorer"] == "cdag_structural"
                    and x["scenario"] == scen
                    and x["seed"] == seed
                )
                psi_vals.append(a[metric])
                cdag_vals.append(b[metric])
            try:
                stat, p = sci_stats.wilcoxon(cdag_vals, psi_vals, alternative="greater")
                stat_table[metric] = {"stat": float(stat), "p": float(p), "n_pairs": len(psi_vals)}
            except ValueError as e:
                stat_table[metric] = {
                    "stat": None,
                    "p": None,
                    "err": str(e),
                    "n_pairs": len(psi_vals),
                }
        summary["wilcoxon_cdag_gt_psi"] = stat_table

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        mlflow.log_artifact(args.out)

        # Human-readable table.
        print("\n=== Phase 3 H1: attribution precision (CDAG vs PSI) ===")
        print(f"{'scorer':<20}{'top1_acc':>18}{'top3_acc':>18}{'MRR':>18}{'AUROC':>18}")
        for name in ("psi_stub", "cdag_structural"):
            a = summary["aggregates"][name]
            print(
                f"{name:<20}"
                f"{a['top1_acc'][0]:>10.4f} ± {a['top1_acc'][1]:.4f}"
                f"{a['top3_acc'][0]:>10.4f} ± {a['top3_acc'][1]:.4f}"
                f"{a['mean_reciprocal_rank'][0]:>10.4f} ± {a['mean_reciprocal_rank'][1]:.4f}"
                f"{a['auroc'][0]:>10.4f} ± {a['auroc'][1]:.4f}"
            )
        print("\nWilcoxon signed-rank, CDAG > PSI:")
        for metric, v in stat_table.items():
            print(f"  {metric}: stat={v.get('stat')} p={v.get('p')} n_pairs={v.get('n_pairs')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
