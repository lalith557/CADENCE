"""Phase 1 gate runner.

Trains FraudNet on Credit Card Fraud, then runs three baselines
(PSI+full-retrain, fixed-schedule, EWC-only) across ≥5 seeds and one anchor
drift scenario. Records aggregated results to MLflow and prints a table
suitable for pasting into docs/results.md.

Usage:
    python -m benchmarks.phase1_run --config configs/default.yaml --seeds 5
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path

import mlflow
import numpy as np
from scipy import stats as sci_stats

from benchmarks.baselines import (
    BaselineOutcome,
    BaselineRunConfig,
    EWCOnlyStrategy,
    FixedScheduleStrategy,
    PSIFullRetrainStrategy,
    run_baseline,
)
from benchmarks.synthetic_drift_gen import build_default_scenarios
from cadence.adapters.neural import FraudNet, FraudNetConfig
from cadence.common.config import load_config
from cadence.common.logging import get_logger
from cadence.common.seeds import set_global_seed
from cadence.common.tracking import start_run
from cadence.data.loaders import load_credit_card_fraud

log = get_logger("cadence.benchmarks.phase1")


def _train_baseline_adapter(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    model_cfg,
) -> FraudNet:
    adapter = FraudNet(
        FraudNetConfig(
            input_dim=input_dim,
            hidden_dims=list(model_cfg.hidden_dims),
            dropout=model_cfg.dropout,
            lr=model_cfg.lr,
            batch_size=model_cfg.batch_size,
            max_epochs=model_cfg.max_epochs,
            early_stopping_patience=model_cfg.early_stopping_patience,
            class_weighted=model_cfg.class_weighted,
        )
    )
    result = adapter.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    log.info(
        "baseline_train_done",
        best_val_f1=result.best_val_f1,
        epochs_run=result.epochs_run,
        wall_s=result.total_wall_s,
    )
    return adapter


def _aggregate(outcomes: Sequence[BaselineOutcome]) -> dict:
    def _mean_std(values):
        vs = list(values)
        if not vs:
            return (0.0, 0.0)
        return (float(statistics.fmean(vs)), float(statistics.pstdev(vs)) if len(vs) > 1 else 0.0)

    return {
        "n_seeds": len(outcomes),
        "mean_f1": _mean_std(o.mean_f1 for o in outcomes),
        "min_f1": _mean_std(o.min_f1 for o in outcomes),
        "gpu_hr": _mean_std(o.total_gpu_seconds / 3600.0 for o in outcomes),
        "kg_co2": _mean_std(o.total_kg_co2 for o in outcomes),
        "sla_windows_below": _mean_std(o.sla_windows_below for o in outcomes),
        "forgetting": _mean_std(o.forgetting_score for o in outcomes),
        "sla_recovery_windows": _mean_std(
            (o.sla_recovery_windows if o.sla_recovery_windows is not None else -1) for o in outcomes
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--scenario", default="amount_multiplicative_abrupt")
    p.add_argument("--n-windows", type=int, default=15)
    p.add_argument("--window-size", type=int, default=2048)
    p.add_argument("--sla", type=float, default=0.90)
    p.add_argument("--out", default="experiments/phase1_summary.json")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    log.info("phase1_start", config=args.config, seeds=args.seeds, scenario=args.scenario)

    # 1) Load Credit Card Fraud once, then split off:
    #    * a training pool (used to pretrain FraudNet, also the historical replay buffer)
    #    * a "production stream" pool (used as the future traffic that drifts)
    #    * an "unrelated held-out" pool (never touched by retrains — measures forgetting)
    ds = load_credit_card_fraud(cfg.data, seed=42)
    n_train = ds.X_train.shape[0]
    # Split X_train into (pretrain 70%, stream 25%, unrelated 5%).
    idx = np.arange(n_train)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    n_pretrain = int(0.70 * n_train)
    n_stream = int(0.25 * n_train)
    pretrain_idx = idx[:n_pretrain]
    stream_idx = idx[n_pretrain : n_pretrain + n_stream]
    unrelated_idx = idx[n_pretrain + n_stream :]

    X_pre = ds.X_train[pretrain_idx]
    y_pre = ds.y_train[pretrain_idx]
    X_stream = ds.X_train[stream_idx]
    y_stream = ds.y_train[stream_idx]
    X_unrel = ds.X_train[unrelated_idx]
    y_unrel = ds.y_train[unrelated_idx]

    # 2) Train the baseline FraudNet once (deterministically), snapshot state.
    set_global_seed(42)
    adapter = _train_baseline_adapter(
        X_pre[:-5000], y_pre[:-5000], X_pre[-5000:], y_pre[-5000:], ds.X_train.shape[1], cfg.model
    )
    baseline_state = adapter.state_dict()
    baseline_threshold = adapter.decision_threshold

    # Log baseline (undrifted) metrics — this is R-1's honest baseline number.
    from sklearn.metrics import f1_score, roc_auc_score

    probs = adapter.predict_proba(ds.X_test)
    preds = (probs >= baseline_threshold).astype(np.int64)
    undrifted_f1 = float(f1_score(ds.y_test, preds, zero_division=0))
    undrifted_auroc = float(roc_auc_score(ds.y_test, probs))
    log.info(
        "baseline_undrifted",
        undrifted_test_f1=undrifted_f1,
        undrifted_test_auroc=undrifted_auroc,
        tuned_threshold=baseline_threshold,
    )

    # 3) Pick the drift scenario.
    all_scenarios = build_default_scenarios(ds.feature_names)
    scenario = next(s for s in all_scenarios if s.name == args.scenario)

    # 4) Run each strategy × seed.
    run_cfg = BaselineRunConfig(
        window_size=args.window_size,
        n_windows=args.n_windows,
        sla_target=args.sla,
    )
    strategy_factories = [PSIFullRetrainStrategy, FixedScheduleStrategy, EWCOnlyStrategy]
    strategy_names = [f().name for f in strategy_factories]

    outcomes: dict[str, list[BaselineOutcome]] = {name: [] for name in strategy_names}

    with start_run(cfg, run_name=f"phase1-{args.scenario}") as parent_run:
        mlflow.log_param("phase", 1)
        mlflow.log_param("scenario", args.scenario)
        mlflow.log_param("n_seeds", args.seeds)
        mlflow.log_param("window_size", args.window_size)
        mlflow.log_param("n_windows", args.n_windows)
        mlflow.log_param("sla_target", args.sla)
        mlflow.log_metric("undrifted_test_f1", undrifted_f1)
        mlflow.log_metric("undrifted_test_auroc", undrifted_auroc)
        mlflow.log_metric("tuned_threshold", baseline_threshold)

        for factory in strategy_factories:
            for seed in range(args.seeds):
                set_global_seed(seed)
                strategy = factory()  # fresh instance per seed — W-19
                outcome = run_baseline(
                    adapter=adapter,
                    baseline_state=baseline_state,
                    baseline_threshold=baseline_threshold,
                    historical_X=X_pre,
                    historical_y=y_pre,
                    stream_X_raw=X_stream,
                    stream_y_raw=y_stream,
                    unrelated_X=X_unrel,
                    unrelated_y=y_unrel,
                    scenario=scenario,
                    strategy=strategy,
                    seed=seed,
                    cfg=run_cfg,
                    feature_names=ds.feature_names,
                )
                outcomes[strategy.name].append(outcome)

                # per-seed metrics to MLflow
                mlflow.log_metric(f"{strategy.name}/mean_f1_seed{seed}", outcome.mean_f1)
                mlflow.log_metric(
                    f"{strategy.name}/gpu_hr_seed{seed}", outcome.total_gpu_seconds / 3600.0
                )
                mlflow.log_metric(f"{strategy.name}/kg_co2_seed{seed}", outcome.total_kg_co2)
                mlflow.log_metric(
                    f"{strategy.name}/forgetting_seed{seed}", outcome.forgetting_score
                )

        # 5) Aggregate + paired stats.
        agg = {name: _aggregate(o) for name, o in outcomes.items()}
        summary_json = {"scenario": args.scenario, "aggregates": {}}
        for name, a in agg.items():
            summary_json["aggregates"][name] = {k: v for k, v in a.items()}
            for k, v in a.items():
                if isinstance(v, tuple):
                    mean, std = v
                    mlflow.log_metric(f"{name}/{k}/mean", mean)
                    mlflow.log_metric(f"{name}/{k}/std", std)

        # Paired Wilcoxon on mean_f1 between baselines.
        stats_table = {}
        for i, a in enumerate(strategy_names):
            for b in strategy_names[i + 1 :]:
                pairs_a = [o.mean_f1 for o in outcomes[a]]
                pairs_b = [o.mean_f1 for o in outcomes[b]]
                if len(pairs_a) < 2 or len(pairs_b) < 2:
                    continue
                try:
                    stat, p = sci_stats.wilcoxon(pairs_a, pairs_b)
                    stats_table[f"{a}_vs_{b}_mean_f1"] = {"stat": float(stat), "p": float(p)}
                except ValueError:
                    stats_table[f"{a}_vs_{b}_mean_f1"] = {"stat": None, "p": None}
        summary_json["wilcoxon"] = stats_table

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary_json, f, indent=2, default=str)
        mlflow.log_artifact(args.out)

        # Human-readable table.
        print("\n=== Phase 1 baseline results (scenario:", args.scenario, ") ===")
        header = ["strategy", "mean_f1", "min_f1", "gpu_hr", "kg_co2", "sla_below", "forgetting"]
        print(" | ".join(f"{h:>18}" for h in header))
        for name, a in agg.items():
            row = [
                name,
                f"{a['mean_f1'][0]:.4f} ± {a['mean_f1'][1]:.4f}",
                f"{a['min_f1'][0]:.4f} ± {a['min_f1'][1]:.4f}",
                f"{a['gpu_hr'][0]:.4f} ± {a['gpu_hr'][1]:.4f}",
                f"{a['kg_co2'][0]:.4f} ± {a['kg_co2'][1]:.4f}",
                f"{a['sla_windows_below'][0]:.2f}",
                f"{a['forgetting'][0]:+.4f}",
            ]
            print(" | ".join(f"{c:>18}" for c in row))
        print("\nWilcoxon (mean_f1):")
        for k, v in stats_table.items():
            print(f"  {k}: stat={v['stat']}, p={v['p']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
