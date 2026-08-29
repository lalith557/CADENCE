"""Gate D runner — Step D of the Execution Plan (H3 forgetting).

Compares the forgetting rate of a Part-3B partial retrain (EWC-regularized
Layer 1 fine-tune) against a naive full retrain, both measured on the
genuinely disjoint sub-population defined by `cadence.data.disjoint`
(fixes W-21).

For each seed:
  1. Pretrain FraudNet on the main (non-disjoint) training slice.
  2. Snapshot baseline F1 on the disjoint sub-population.
  3. Inject drift into the main stream (amount_multiplicative_abrupt by default).
  4. Path A: run partial retrain (RetrainExecutor with EWC on Layer 1).
             Measure post_partial_disjoint_F1.
  5. Path B: reset the adapter to baseline state, run naive full retrain.
             Measure post_full_disjoint_F1.
  6. forgetting_partial = baseline_disjoint_F1 - post_partial_disjoint_F1
     forgetting_full    = baseline_disjoint_F1 - post_full_disjoint_F1
  7. Paired Wilcoxon: forgetting_partial < forgetting_full (i.e. partial
     preserves the sub-population better). p < 0.05 -> H3 supported.

Usage:
    python -m benchmarks.phase_d_run --seeds 10 --sla 0.65 \
        --window-size 2048 --disjoint-criterion high_amount
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import mlflow
import numpy as np
from scipy import stats as sci_stats
from sklearn.metrics import f1_score

from benchmarks.baselines.harness import make_baseline_stream
from benchmarks.synthetic_drift_gen import build_default_scenarios
from cadence.adapters.neural import FraudNet, FraudNetConfig
from cadence.common.config import load_config
from cadence.common.device import log_device_info
from cadence.common.logging import get_logger
from cadence.common.seeds import set_global_seed
from cadence.common.tracking import start_run
from cadence.data.disjoint import make_disjoint_split
from cadence.data.loaders import load_credit_card_fraud
from cadence.executor import ExecutorConfig, RetrainExecutor

log = get_logger("cadence.benchmarks.phase_d")


def _mean_std(vs):
    vs = list(vs)
    if not vs:
        return (0.0, 0.0)
    return (
        float(statistics.fmean(vs)),
        float(statistics.pstdev(vs)) if len(vs) > 1 else 0.0,
    )


def _f1(adapter, X, y) -> float:
    if X.size == 0:
        return 0.0
    probs = adapter.predict_proba(X)
    preds = (probs >= adapter.decision_threshold).astype(np.int64)
    return float(f1_score(y, preds, zero_division=0))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--scenario", default="amount_multiplicative_abrupt")
    p.add_argument("--window-size", type=int, default=2048)
    p.add_argument("--n-windows", type=int, default=3)
    p.add_argument("--sla", type=float, default=0.65)
    p.add_argument("--disjoint-criterion", default="high_amount",
                   choices=["high_amount", "late_time", "class_positive"])
    p.add_argument("--finetune-epochs", type=int, default=3)
    p.add_argument("--fullretrain-epochs", type=int, default=8)
    p.add_argument("--target-layer", default="layer1")
    p.add_argument("--out", default="experiments/phase_d_summary.json")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    log_device_info()
    log.info(
        "phase_d_start",
        seeds=args.seeds,
        scenario=args.scenario,
        criterion=args.disjoint_criterion,
    )

    # 1) Load Fraud + split so the disjoint segment is genuinely held out.
    ds = load_credit_card_fraud(cfg.data, seed=42)
    split = make_disjoint_split(
        ds.X_train,
        ds.y_train,
        ds.feature_names,
        criterion=args.disjoint_criterion,
        train_frac=0.70,
        stream_frac=0.25,
        seed=42,
    )
    log.info(
        "disjoint_split_built",
        criterion=split.criterion,
        n_disjoint=split.n_disjoint,
        disjoint_positive_rate=split.disjoint_positive_rate,
        n_main_train=int(split.main_train_X.shape[0]),
        n_main_stream=int(split.main_stream_X.shape[0]),
    )

    X_pre = split.main_train_X
    y_pre = split.main_train_y
    X_stream = split.main_stream_X
    y_stream = split.main_stream_y
    X_dis = split.disjoint_X
    y_dis = split.disjoint_y

    # 2) Pretrain FraudNet ONCE (deterministic seed=42) — reused as baseline
    # for every partial-vs-full pair.
    set_global_seed(42)
    adapter_template = FraudNet(
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
    adapter_template.fit(X_pre[:-5000], y_pre[:-5000], X_val=X_pre[-5000:], y_val=y_pre[-5000:])
    baseline_state = adapter_template.state_dict()
    baseline_threshold = adapter_template.decision_threshold
    baseline_disjoint_f1 = _f1(adapter_template, X_dis, y_dis)
    log.info(
        "baseline_ready",
        baseline_disjoint_f1=baseline_disjoint_f1,
        baseline_threshold=baseline_threshold,
    )

    scenarios = build_default_scenarios(ds.feature_names)
    scenario = next(s for s in scenarios if s.name == args.scenario)

    forgetting_partial: list[float] = []
    forgetting_full: list[float] = []
    post_partial_f1: list[float] = []
    post_full_f1: list[float] = []
    rollback_partial = 0
    rollback_full = 0
    n_use = args.window_size * args.n_windows

    with start_run(cfg, run_name=f"phase_d_h3_{args.disjoint_criterion}"):
        mlflow.log_param("phase", "D")
        mlflow.log_param("scenario", args.scenario)
        mlflow.log_param("disjoint_criterion", args.disjoint_criterion)
        mlflow.log_param("seeds", args.seeds)
        mlflow.log_param("finetune_epochs", args.finetune_epochs)
        mlflow.log_param("fullretrain_epochs", args.fullretrain_epochs)
        mlflow.log_param("target_layer", args.target_layer)
        mlflow.log_metric("baseline_disjoint_f1", baseline_disjoint_f1)
        mlflow.log_metric("n_disjoint", split.n_disjoint)

        for seed in range(args.seeds):
            set_global_seed(seed)
            stream_X, stream_y, _ = make_baseline_stream(X_stream, y_stream, scenario, seed=seed)
            window_X = stream_X[:n_use]
            window_y = stream_y[:n_use]

            # ---------- Path A: PARTIAL retrain (Part-3B EWC) ----------
            adapter_p = adapter_template.clone()
            adapter_p.load_state_dict(baseline_state)
            adapter_p.decision_threshold = baseline_threshold
            exec_p = RetrainExecutor(
                adapter_p,
                historical_X=X_pre,
                historical_y=y_pre,
                cfg=ExecutorConfig(
                    sla_target=args.sla,
                    finetune_epochs=args.finetune_epochs,
                    fullretrain_epochs=args.fullretrain_epochs,
                ),
            )
            report_p = exec_p.execute(
                action="partial",
                window_X=window_X,
                window_y=window_y,
                target_layer=args.target_layer,
            )
            post_p_dis = _f1(adapter_p, X_dis, y_dis)
            forget_p = baseline_disjoint_f1 - post_p_dis

            # ---------- Path B: FULL retrain (naive) ----------
            adapter_f = adapter_template.clone()
            adapter_f.load_state_dict(baseline_state)
            adapter_f.decision_threshold = baseline_threshold
            exec_f = RetrainExecutor(
                adapter_f,
                historical_X=X_pre,
                historical_y=y_pre,
                cfg=ExecutorConfig(
                    sla_target=args.sla,
                    finetune_epochs=args.finetune_epochs,
                    fullretrain_epochs=args.fullretrain_epochs,
                ),
            )
            report_f = exec_f.execute(
                action="full",
                window_X=window_X,
                window_y=window_y,
            )
            post_f_dis = _f1(adapter_f, X_dis, y_dis)
            forget_f = baseline_disjoint_f1 - post_f_dis

            forgetting_partial.append(forget_p)
            forgetting_full.append(forget_f)
            post_partial_f1.append(post_p_dis)
            post_full_f1.append(post_f_dis)
            rollback_partial += int(report_p.event.was_rolled_back)
            rollback_full += int(report_f.event.was_rolled_back)

            log.info(
                "seed_done",
                seed=seed,
                forget_partial=forget_p,
                forget_full=forget_f,
                post_partial_disjoint_f1=post_p_dis,
                post_full_disjoint_f1=post_f_dis,
                partial_rolled_back=report_p.event.was_rolled_back,
                full_rolled_back=report_f.event.was_rolled_back,
            )
            mlflow.log_metric(f"forget_partial_seed{seed}", forget_p)
            mlflow.log_metric(f"forget_full_seed{seed}", forget_f)

        # Paired Wilcoxon: partial < full (lower forgetting = better).
        try:
            stat, p_val = sci_stats.wilcoxon(
                forgetting_partial, forgetting_full, alternative="less"
            )
            wilcoxon = {"stat": float(stat), "p": float(p_val), "n_pairs": len(forgetting_partial)}
        except ValueError as e:
            wilcoxon = {
                "stat": None,
                "p": None,
                "err": str(e),
                "n_pairs": len(forgetting_partial),
            }

        summary = {
            "config": vars(args),
            "baseline_disjoint_f1": baseline_disjoint_f1,
            "n_disjoint": split.n_disjoint,
            "disjoint_positive_rate": split.disjoint_positive_rate,
            "forgetting_partial": {
                "mean": _mean_std(forgetting_partial)[0],
                "std": _mean_std(forgetting_partial)[1],
                "values": forgetting_partial,
            },
            "forgetting_full": {
                "mean": _mean_std(forgetting_full)[0],
                "std": _mean_std(forgetting_full)[1],
                "values": forgetting_full,
            },
            "post_partial_disjoint_f1": {
                "mean": _mean_std(post_partial_f1)[0],
                "std": _mean_std(post_partial_f1)[1],
            },
            "post_full_disjoint_f1": {
                "mean": _mean_std(post_full_f1)[0],
                "std": _mean_std(post_full_f1)[1],
            },
            "rollback_partial": rollback_partial,
            "rollback_full": rollback_full,
            "wilcoxon_partial_less_full": wilcoxon,
        }

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        mlflow.log_artifact(args.out)
        mlflow.log_metric(
            "forgetting_partial_mean", summary["forgetting_partial"]["mean"]
        )
        mlflow.log_metric("forgetting_full_mean", summary["forgetting_full"]["mean"])
        if wilcoxon.get("p") is not None:
            mlflow.log_metric("wilcoxon_p_partial_less_full", wilcoxon["p"])

        print("\n=== Gate D: H3 forgetting on disjoint segment ===")
        print(f"Disjoint criterion: {args.disjoint_criterion}  "
              f"n={split.n_disjoint}  positive_rate={split.disjoint_positive_rate:.4f}")
        print(f"Baseline disjoint F1: {baseline_disjoint_f1:.4f}")
        print(
            f"Forgetting (partial): {summary['forgetting_partial']['mean']:+.4f} "
            f"+/- {summary['forgetting_partial']['std']:.4f}"
        )
        print(
            f"Forgetting (full)   : {summary['forgetting_full']['mean']:+.4f} "
            f"+/- {summary['forgetting_full']['std']:.4f}"
        )
        print(
            f"Post-partial disjoint F1: {summary['post_partial_disjoint_f1']['mean']:.4f} "
            f"+/- {summary['post_partial_disjoint_f1']['std']:.4f}"
        )
        print(
            f"Post-full disjoint F1   : {summary['post_full_disjoint_f1']['mean']:.4f} "
            f"+/- {summary['post_full_disjoint_f1']['std']:.4f}"
        )
        print(f"Rollbacks: partial={rollback_partial}  full={rollback_full}")
        print(
            f"Wilcoxon partial<full: stat={wilcoxon.get('stat')} "
            f"p={wilcoxon.get('p')}  n_pairs={wilcoxon.get('n_pairs')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
