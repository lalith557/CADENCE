"""Phase F, part 1: LightGBM tree adapter generality on Give Me Some Credit.

Runs the same three-way comparison as `phase_e_run`:
  * cadence_rule     — executor + escalation ladder + false-alarm suppression
  * periodic         — retrain every N windows
  * reactive_full    — PSI + full retrain

But swaps the neural FraudNet for a LightGBM booster (`LightGBMAdapter`).
Because trees have no layer-level partial retrain, the executor's partial
path immediately raises NotImplementedError and the ladder falls through
to full — which is the *correct* behaviour under §4's fallback design.

Gate F contribution: shows CADENCE's decision loop is *architecture-agnostic*
(the whole point of the ModelAdapter interface).

Usage:
    python -m benchmarks.phase_f_tree --seeds 3 --n-windows 8 --sla 0.30
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import f1_score

from benchmarks.baselines.harness import make_baseline_stream
from benchmarks.synthetic_drift_gen import DriftScenario, DriftSchedule, FeatureShiftSpec
from cadence.adapters.tree import LightGBMAdapter, LightGBMConfig
from cadence.collector.drift_trigger import DriftTriggerConfig, PSITrigger
from cadence.common.config import load_config
from cadence.common.device import log_device_info
from cadence.common.logging import get_logger
from cadence.common.seeds import set_global_seed
from cadence.common.tracking import start_run
from cadence.data.loaders import load_give_me_some_credit
from cadence.executor import (
    EscalationConfig,
    EscalationDecision,
    EscalationLadder,
    ExecutorConfig,
    RetrainExecutor,
)
from cadence.executor.fallbacks import FalseAlarmConfig, is_false_alarm

log = get_logger("cadence.benchmarks.phase_f_tree")


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


def _run(
    *,
    adapter: LightGBMAdapter,
    baseline_state: dict,
    stream_X: np.ndarray,
    stream_y: np.ndarray,
    trigger: PSITrigger,
    strategy_name: str,
    sla_target: float,
    window_size: int,
    n_windows: int,
    executor_cfg: ExecutorConfig,
    historical_X: np.ndarray,
    historical_y: np.ndarray,
    periodic_period: int = 5,
) -> dict:
    adapter.load_state_dict(baseline_state)
    exec_ = RetrainExecutor(
        adapter,
        historical_X=historical_X,
        historical_y=historical_y,
        cfg=executor_cfg,
    )
    ladder = EscalationLadder(EscalationConfig(ladder=("partial", "full"), unrecoverable_streak=3))

    rolling_f1: list[float] = []
    action_counts = {"no-op": 0, "partial": 0, "full": 0}
    n_rollbacks = 0
    n_escalated = 0
    n_periodic_since = 0
    partial_fallbacks = 0  # times partial NotImplementedError -> full

    for w in range(n_windows):
        start = w * window_size
        end = min(start + window_size, stream_X.shape[0])
        if end <= start:
            break
        window_X = stream_X[start:end]
        window_y = stream_y[start:end]
        pre_f1 = _f1(adapter, window_X, window_y)
        rolling_f1.append(pre_f1)

        if strategy_name == "cadence_rule":
            alert = trigger.evaluate(window_X)
            if is_false_alarm(alert.fired, pre_f1, cfg=FalseAlarmConfig(sla_target=sla_target)):
                action_counts["no-op"] += 1
                continue
            if pre_f1 >= sla_target:
                action_counts["no-op"] += 1
                continue
            # Trees have no tap layers → partial raises NotImplementedError →
            # executor logs adapter_no_partial_fit_fallback_full and does full.
            # The ladder still tries partial first; the executor's fallback
            # inside partial_fit path turns it into a full.
            try:
                result = ladder.run_ladder(
                    exec_,
                    window_X=window_X,
                    window_y=window_y,
                    # target_layer=None would raise on partial; pick a stub
                    # so the executor's own fallback catches the NIE.
                    target_layer_for_partial="tree_stub",
                )
            except NotImplementedError:
                # ladder didn't wrap; still fall through to full manually
                result = None
                partial_fallbacks += 1
                report = exec_.execute(
                    action="full", window_X=window_X, window_y=window_y
                )
                action_counts["full"] += 1
                if report.event.was_rolled_back:
                    n_rollbacks += 1
                continue

            for report in result.reports:
                action_counts[report.action_taken] += 1
                if report.event.was_rolled_back:
                    n_rollbacks += 1
            if result.decision == EscalationDecision.ESCALATED_HUMAN:
                n_escalated += 1

        elif strategy_name == "periodic":
            n_periodic_since += 1
            if n_periodic_since >= periodic_period:
                report = exec_.execute(
                    action="full", window_X=window_X, window_y=window_y
                )
                action_counts["full"] += 1
                if report.event.was_rolled_back:
                    n_rollbacks += 1
                n_periodic_since = 0
            else:
                action_counts["no-op"] += 1

        elif strategy_name == "reactive_full":
            alert = trigger.evaluate(window_X)
            if alert.fired:
                report = exec_.execute(
                    action="full", window_X=window_X, window_y=window_y
                )
                action_counts["full"] += 1
                if report.event.was_rolled_back:
                    n_rollbacks += 1
            else:
                action_counts["no-op"] += 1
        else:
            raise ValueError(strategy_name)

    return {
        "strategy": strategy_name,
        "mean_f1": float(np.mean(rolling_f1)) if rolling_f1 else 0.0,
        "min_f1": float(np.min(rolling_f1)) if rolling_f1 else 0.0,
        "action_counts": action_counts,
        "n_rollbacks": n_rollbacks,
        "n_escalated_human": n_escalated,
        "partial_fallbacks": partial_fallbacks,
        "rolling_f1": rolling_f1,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--window-size", type=int, default=1024)
    p.add_argument("--n-windows", type=int, default=8)
    p.add_argument("--sla", type=float, default=0.30)
    p.add_argument("--periodic-period", type=int, default=4)
    p.add_argument("--fullretrain-epochs", type=int, default=100)  # boosting rounds
    p.add_argument("--out", default="experiments/phase_f_tree.json")
    p.add_argument("--n-estimators", type=int, default=100)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    log_device_info()
    log.info("phase_f_tree_start", seeds=args.seeds)

    ds = load_give_me_some_credit(cfg.data, seed=42)
    # Split like Phase 1: pretrain / stream / (no unrelated here).
    n = ds.X_train.shape[0]
    rng = np.random.default_rng(42)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_pre = int(0.70 * n)
    X_pre = ds.X_train[idx[:n_pre]]
    y_pre = ds.y_train[idx[:n_pre]]
    X_stream = ds.X_train[idx[n_pre:]]
    y_stream = ds.y_train[idx[n_pre:]]

    # Pretrain once at seed 42.
    set_global_seed(42)
    adapter = LightGBMAdapter(
        LightGBMConfig(n_estimators=args.n_estimators, num_leaves=31, seed=42)
    )
    adapter.fit(X_pre[:-2000], y_pre[:-2000], X_val=X_pre[-2000:], y_val=y_pre[-2000:])
    baseline_state = adapter.state_dict()

    baseline_f1 = _f1(adapter, X_pre[-2000:], y_pre[-2000:])
    log.info("baseline_ready", baseline_f1=baseline_f1, sla_target=args.sla)

    # Inject amount-like drift on a high-signal GMSC feature: shift
    # DebtRatio upward by 2x on the stream. Preserves temporal-order but
    # guarantees a meaningful degradation to test the loop against.
    drift_feature_idx = ds.feature_names.index("DebtRatio") if "DebtRatio" in ds.feature_names else 0
    scenario = DriftScenario(
        name="gmsc_debt_ratio_x2",
        spec=FeatureShiftSpec(feature_idx=drift_feature_idx, multiplicative=2.0, additive=0.0),
        schedule=DriftSchedule.ABRUPT,
        start_step=0,
        full_effect_step=0,
        feature_names=ds.feature_names,
    )
    drifted_X, drifted_y, _ = make_baseline_stream(X_stream, y_stream, scenario, seed=0)

    strategies = ("cadence_rule", "periodic", "reactive_full")
    results: dict[str, list[dict]] = {s: [] for s in strategies}
    exec_cfg = ExecutorConfig(
        sla_target=args.sla,
        finetune_epochs=args.fullretrain_epochs // 4,
        fullretrain_epochs=args.fullretrain_epochs,
    )

    with start_run(cfg, run_name="phase_f_tree_gmsc"):
        mlflow.log_param("phase", "F/tree")
        mlflow.log_param("adapter_family", adapter.family)
        mlflow.log_param("dataset", "give_me_some_credit")
        mlflow.log_metric("baseline_pretrain_f1", baseline_f1)

        for seed in range(args.seeds):
            set_global_seed(seed)
            trigger = PSITrigger(
                baseline_X=X_pre,
                feature_names=ds.feature_names,
                cfg=DriftTriggerConfig(psi_threshold=0.25, min_window_rows=200),
            )
            for strat_name in strategies:
                t0 = time.perf_counter()
                out = _run(
                    adapter=adapter,
                    baseline_state=baseline_state,
                    stream_X=drifted_X,
                    stream_y=drifted_y,
                    trigger=trigger,
                    strategy_name=strat_name,
                    sla_target=args.sla,
                    window_size=args.window_size,
                    n_windows=args.n_windows,
                    executor_cfg=exec_cfg,
                    historical_X=X_pre,
                    historical_y=y_pre,
                    periodic_period=args.periodic_period,
                )
                out["wall_s"] = time.perf_counter() - t0
                out["seed"] = seed
                results[strat_name].append(out)
                log.info(
                    "strategy_done",
                    strategy=strat_name,
                    seed=seed,
                    mean_f1=out["mean_f1"],
                    actions=out["action_counts"],
                    n_rollbacks=out["n_rollbacks"],
                )
                mlflow.log_metric(f"{strat_name}/mean_f1_seed{seed}", out["mean_f1"])
                gc.collect()

        def _agg(name):
            r = results[name]
            return {
                "n_seeds": len(r),
                "mean_f1": _mean_std(x["mean_f1"] for x in r),
                "min_f1": _mean_std(x["min_f1"] for x in r),
                "actions_full": _mean_std(x["action_counts"]["full"] for x in r)[0],
                "actions_partial": _mean_std(x["action_counts"]["partial"] for x in r)[0],
                "actions_noop": _mean_std(x["action_counts"]["no-op"] for x in r)[0],
                "rollbacks_mean": _mean_std(x["n_rollbacks"] for x in r)[0],
            }

        summary = {
            "adapter_family": adapter.family,
            "dataset": ds.summary(),
            "baseline_pretrain_f1": baseline_f1,
            "scenario": scenario.name,
            "aggregates": {s: _agg(s) for s in strategies},
            "config": vars(args),
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        mlflow.log_artifact(args.out)

        print(f"\n=== Gate F (tree): LightGBM on GMSC + {scenario.name} ===")
        print(f"Baseline F1: {baseline_f1:.4f}  SLA: {args.sla}")
        print(f"{'strategy':<18}{'mean_f1':>18}{'min_f1':>18}{'no-op':>10}{'partial':>10}{'full':>10}{'rollbacks':>12}")
        for name, a in summary["aggregates"].items():
            print(
                f"{name:<18}"
                f"{a['mean_f1'][0]:>10.4f} +/- {a['mean_f1'][1]:.4f}"
                f"{a['min_f1'][0]:>10.4f} +/- {a['min_f1'][1]:.4f}"
                f"{a['actions_noop']:>10.1f}"
                f"{a['actions_partial']:>10.1f}"
                f"{a['actions_full']:>10.1f}"
                f"{a['rollbacks_mean']:>12.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
