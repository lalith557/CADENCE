"""Gate E runner — Step E of the Execution Plan.

Real-drift end-to-end on Elec2 and Airlines. This is the "outcome graded"
part of H2: unlike Steps A/B/C/D, there is NO injected drift and NO ground-
truth root cause. The stream is a natural temporal shift; the model
degrades on its own; we grade each strategy by whether it recovers F1 and
at what cost.

Three strategies compared per dataset × seed:
  * cadence_rule    — CADENCE with the executor+fallbacks driven by a
                      simple rule policy: if window F1 < SLA, run partial
                      retrain of the first tap layer; if that rolls back,
                      escalation ladder falls through to full retrain
                      (§4.3). Also does §4.6 false-alarm suppression via
                      the PSI+delayed-F1 check.
  * periodic        — full retrain every N windows regardless of drift.
  * reactive_full   — Phase 1 baseline: PSI alert -> full retrain.

Metrics per (dataset, strategy, seed):
  * mean F1 across all windows
  * min F1
  * total GPU-hr (closed-form cost model)
  * total kg CO2
  * n_actions {no-op, partial, full}

Grade (Gate E, per plan): CADENCE recovers performance at lower total cost
than periodic and reactive-full. Reported honestly — including where it
doesn't.

Usage:
    python -m benchmarks.phase_e_run --dataset elec2 --seeds 3
    python -m benchmarks.phase_e_run --dataset airlines --seeds 3
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
import torch
from sklearn.metrics import f1_score

from cadence.adapters.neural import FraudNet, FraudNetConfig
from cadence.carbon.model import GridProfile, HardwareProfile, estimate_cost
from cadence.collector.drift_trigger import DriftTriggerConfig, PSITrigger
from cadence.common.config import load_config
from cadence.common.device import cuda_memory_snapshot, log_device_info
from cadence.common.logging import get_logger
from cadence.common.seeds import set_global_seed
from cadence.common.tracking import start_run
from cadence.data.realdrift import load_airlines, load_elec2
from cadence.executor import (
    EscalationConfig,
    EscalationDecision,
    EscalationLadder,
    ExecutorConfig,
    RetrainExecutor,
)
from cadence.executor.fallbacks import FalseAlarmConfig, is_false_alarm

log = get_logger("cadence.benchmarks.phase_e")


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


def _build_adapter(input_dim: int, cfg) -> FraudNet:
    return FraudNet(
        FraudNetConfig(
            input_dim=input_dim,
            hidden_dims=list(cfg.model.hidden_dims),
            dropout=cfg.model.dropout,
            lr=cfg.model.lr,
            batch_size=cfg.model.batch_size,
            max_epochs=cfg.model.max_epochs,
            early_stopping_patience=cfg.model.early_stopping_patience,
            class_weighted=cfg.model.class_weighted,
        )
    )


def _pretrain(
    input_dim: int, cfg, X_train: np.ndarray, y_train: np.ndarray, seed: int
) -> tuple[FraudNet, dict, float, float]:
    set_global_seed(seed)
    adapter = _build_adapter(input_dim, cfg)
    n_val = max(500, int(0.10 * X_train.shape[0]))
    adapter.fit(
        X_train[:-n_val],
        y_train[:-n_val],
        X_val=X_train[-n_val:],
        y_val=y_train[-n_val:],
    )
    baseline_state = {k: v.clone() for k, v in adapter._module.state_dict().items()}
    baseline_threshold = adapter.decision_threshold
    baseline_f1 = _f1(adapter, X_train[-n_val:], y_train[-n_val:])
    return adapter, baseline_state, baseline_threshold, baseline_f1


def _run_windowed_episode(
    *,
    adapter: FraudNet,
    baseline_state: dict,
    baseline_threshold: float,
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
    target_layer: str = "layer1",
    hardware: HardwareProfile | None = None,
    grid: GridProfile | None = None,
) -> dict:
    hardware = hardware or HardwareProfile()
    grid = grid or GridProfile()

    # Reset adapter to shared baseline for a fair comparison.
    adapter.load_state_dict(baseline_state)
    adapter.decision_threshold = baseline_threshold

    exec_ = RetrainExecutor(
        adapter,
        historical_X=historical_X,
        historical_y=historical_y,
        cfg=executor_cfg,
    )
    ladder = EscalationLadder(EscalationConfig(ladder=("partial", "full"), unrecoverable_streak=3))

    rolling_f1: list[float] = []
    action_counts = {"no-op": 0, "partial": 0, "full": 0}
    total_gpu_seconds = 0.0
    total_kg_co2 = 0.0
    n_rollbacks = 0
    n_escalated = 0
    n_periodic_since = 0

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
            # §4.6 false-alarm suppression using the current-window F1 as the
            # "delayed-label" proxy (labels are always available in this
            # sandbox; in production this would be a real delayed slice).
            alert = trigger.evaluate(window_X)
            if is_false_alarm(alert.fired, pre_f1, cfg=FalseAlarmConfig(sla_target=sla_target)):
                action_counts["no-op"] += 1
                continue
            # If SLA met, no-op. Otherwise ladder: partial → full.
            if pre_f1 >= sla_target:
                action_counts["no-op"] += 1
                continue
            result = ladder.run_ladder(
                exec_,
                window_X=window_X,
                window_y=window_y,
                target_layer_for_partial=target_layer,
            )
            for report in result.reports:
                total_gpu_seconds += report.event.gpu_seconds
                total_kg_co2 += report.event.kg_co2
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
                total_gpu_seconds += report.event.gpu_seconds
                total_kg_co2 += report.event.kg_co2
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
                total_gpu_seconds += report.event.gpu_seconds
                total_kg_co2 += report.event.kg_co2
                action_counts["full"] += 1
                if report.event.was_rolled_back:
                    n_rollbacks += 1
            else:
                action_counts["no-op"] += 1

        else:
            raise ValueError(f"unknown strategy: {strategy_name}")

    return {
        "strategy": strategy_name,
        "rolling_f1": rolling_f1,
        "mean_f1": float(np.mean(rolling_f1)) if rolling_f1 else 0.0,
        "min_f1": float(np.min(rolling_f1)) if rolling_f1 else 0.0,
        "max_f1": float(np.max(rolling_f1)) if rolling_f1 else 0.0,
        "total_gpu_seconds": total_gpu_seconds,
        "total_gpu_hr": total_gpu_seconds / 3600.0,
        "total_kg_co2": total_kg_co2,
        "action_counts": action_counts,
        "n_rollbacks": n_rollbacks,
        "n_escalated_human": n_escalated,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dataset", choices=["elec2", "airlines"], default="elec2")
    p.add_argument("--train-frac", type=float, default=0.30)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--window-size", type=int, default=1024)
    p.add_argument("--n-windows", type=int, default=15)
    p.add_argument("--sla", type=float, default=0.70)
    p.add_argument("--finetune-epochs", type=int, default=3)
    p.add_argument("--fullretrain-epochs", type=int, default=8)
    p.add_argument("--periodic-period", type=int, default=5)
    p.add_argument("--out", default="experiments/phase_e_summary.json")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    log_device_info()
    log.info("phase_e_start", dataset=args.dataset, seeds=args.seeds)

    if args.dataset == "elec2":
        ds = load_elec2(train_frac=args.train_frac)
    else:
        ds = load_airlines(train_frac=args.train_frac)
    log.info("dataset_loaded", **ds.summary())

    X_train, y_train = ds.train_slice()
    X_stream, y_stream = ds.stream_slice()

    # Snapshot pretrain once (deterministic seed=42) so every seed × strategy
    # starts from the same baseline. Per-seed variation lives in the stream
    # sampling and the PSI trigger state.
    adapter, baseline_state, baseline_threshold, baseline_f1 = _pretrain(
        ds.n_features, cfg, X_train, y_train, seed=42
    )
    # Stream-slice F1 with no retraining — how much natural drift there is.
    undrifted_stream_f1 = _f1(
        adapter, X_stream[: args.window_size * args.n_windows], y_stream[: args.window_size * args.n_windows]
    )
    log.info(
        "baseline_ready",
        baseline_f1=baseline_f1,
        undrifted_stream_f1=undrifted_stream_f1,
        sla_target=args.sla,
    )

    strategies = ("cadence_rule", "periodic", "reactive_full")
    results: dict[str, list[dict]] = {s: [] for s in strategies}

    exec_cfg = ExecutorConfig(
        sla_target=args.sla,
        finetune_epochs=args.finetune_epochs,
        fullretrain_epochs=args.fullretrain_epochs,
    )
    hardware = HardwareProfile()
    grid = GridProfile()

    with start_run(cfg, run_name=f"phase_e_{args.dataset}"):
        mlflow.log_param("phase", "E")
        mlflow.log_param("dataset", args.dataset)
        mlflow.log_param("train_frac", args.train_frac)
        mlflow.log_param("seeds", args.seeds)
        mlflow.log_param("window_size", args.window_size)
        mlflow.log_param("n_windows", args.n_windows)
        mlflow.log_param("sla_target", args.sla)
        mlflow.log_param("periodic_period", args.periodic_period)
        mlflow.log_metric("baseline_train_f1", baseline_f1)
        mlflow.log_metric("undrifted_stream_f1", undrifted_stream_f1)

        for seed in range(args.seeds):
            set_global_seed(seed)
            # Fresh PSI trigger per seed (state resets so alerts are independent).
            trigger = PSITrigger(
                baseline_X=X_train,
                feature_names=ds.feature_names,
                cfg=DriftTriggerConfig(psi_threshold=0.25, min_window_rows=200),
            )

            for strat_name in strategies:
                t0 = time.perf_counter()
                out = _run_windowed_episode(
                    adapter=adapter,
                    baseline_state=baseline_state,
                    baseline_threshold=baseline_threshold,
                    stream_X=X_stream,
                    stream_y=y_stream,
                    trigger=trigger,
                    strategy_name=strat_name,
                    sla_target=args.sla,
                    window_size=args.window_size,
                    n_windows=args.n_windows,
                    executor_cfg=exec_cfg,
                    historical_X=X_train,
                    historical_y=y_train,
                    periodic_period=args.periodic_period,
                    target_layer="layer1",
                    hardware=hardware,
                    grid=grid,
                )
                out["wall_s"] = time.perf_counter() - t0
                out["seed"] = seed
                results[strat_name].append(out)
                log.info(
                    "strategy_done",
                    strategy=strat_name,
                    seed=seed,
                    mean_f1=out["mean_f1"],
                    total_gpu_hr=out["total_gpu_hr"],
                    total_kg_co2=out["total_kg_co2"],
                    actions=out["action_counts"],
                    n_rollbacks=out["n_rollbacks"],
                )
                mlflow.log_metric(f"{strat_name}/mean_f1_seed{seed}", out["mean_f1"])
                mlflow.log_metric(f"{strat_name}/gpu_hr_seed{seed}", out["total_gpu_hr"])
                mlflow.log_metric(f"{strat_name}/kg_co2_seed{seed}", out["total_kg_co2"])

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        def _agg(name):
            r = results[name]
            return {
                "n_seeds": len(r),
                "mean_f1": _mean_std(x["mean_f1"] for x in r),
                "min_f1": _mean_std(x["min_f1"] for x in r),
                "gpu_hr": _mean_std(x["total_gpu_hr"] for x in r),
                "kg_co2": _mean_std(x["total_kg_co2"] for x in r),
                "action_counts": {
                    "no-op": _mean_std(x["action_counts"]["no-op"] for x in r),
                    "partial": _mean_std(x["action_counts"]["partial"] for x in r),
                    "full": _mean_std(x["action_counts"]["full"] for x in r),
                },
                "n_rollbacks_mean": _mean_std(x["n_rollbacks"] for x in r)[0],
                "n_escalated_human_mean": _mean_std(x["n_escalated_human"] for x in r)[0],
            }

        summary = {
            "dataset": ds.summary(),
            "baseline_train_f1": baseline_f1,
            "undrifted_stream_f1": undrifted_stream_f1,
            "aggregates": {s: _agg(s) for s in strategies},
            "raw": results,
            "config": vars(args),
            "gpu_memory_end": cuda_memory_snapshot(),
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        mlflow.log_artifact(args.out)

        # Pareto verdict: does CADENCE beat baselines on cost at equal/better F1?
        agg = summary["aggregates"]
        cad = agg["cadence_rule"]
        verdict: dict = {}
        for baseline in ("periodic", "reactive_full"):
            b = agg[baseline]
            verdict[baseline] = {
                "cadence_mean_f1_gap": cad["mean_f1"][0] - b["mean_f1"][0],
                "cadence_gpu_hr_ratio": (
                    cad["gpu_hr"][0] / b["gpu_hr"][0] if b["gpu_hr"][0] > 0 else float("inf")
                ),
                "cadence_kg_co2_ratio": (
                    cad["kg_co2"][0] / b["kg_co2"][0] if b["kg_co2"][0] > 0 else float("inf")
                ),
                "cadence_wins_cost": (
                    cad["gpu_hr"][0] < b["gpu_hr"][0] and cad["mean_f1"][0] >= b["mean_f1"][0]
                ),
            }
        summary["gate_e_verdict"] = verdict
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n=== Gate E: {ds.name} real drift, {args.seeds} seeds ===")
        print(f"Baseline train F1: {baseline_f1:.4f}   Undrifted stream F1: {undrifted_stream_f1:.4f}   SLA: {args.sla}")
        print(
            f"\n{'strategy':<18}{'mean_f1':>18}{'min_f1':>18}{'gpu_hr':>18}{'kg_co2':>18}{'actions {no,part,full}':>26}"
        )
        for name, a in agg.items():
            actions = a["action_counts"]
            print(
                f"{name:<18}"
                f"{a['mean_f1'][0]:>10.4f} +/- {a['mean_f1'][1]:.4f}"
                f"{a['min_f1'][0]:>10.4f} +/- {a['min_f1'][1]:.4f}"
                f"{a['gpu_hr'][0]:>10.6f} +/- {a['gpu_hr'][1]:.6f}"
                f"{a['kg_co2'][0]:>10.6f} +/- {a['kg_co2'][1]:.6f}"
                f"  {{{actions['no-op'][0]:.1f}, {actions['partial'][0]:.1f}, {actions['full'][0]:.1f}}}"
            )
        print("\nGate E verdict:")
        for baseline, v in verdict.items():
            wins = "PASS" if v["cadence_wins_cost"] else "FAIL"
            print(
                f"  cadence vs {baseline}: F1 gap={v['cadence_mean_f1_gap']:+.4f}  "
                f"gpu_hr ratio={v['cadence_gpu_hr_ratio']:.3f}  "
                f"kg_co2 ratio={v['cadence_kg_co2_ratio']:.3f}   [{wins}]"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
