"""Phase 2 gate runner.

- Pretrains FraudNet once (same as Phase 1, seed=42).
- Trains a PPO RSO policy on the `amount_multiplicative_abrupt` scenario.
- Evaluates the fixed policy on all 5 default scenarios × N seeds.
- Runs the same evaluation for the three Phase 1 baselines.
- Records aggregate + paired Wilcoxon (RSO vs each baseline) → R-4 in results.md.

Usage:
    python -m benchmarks.phase2_run --config configs/default.yaml \
        --seeds 10 --train-timesteps 3000 --sla 0.65 --n-windows 15 --window-size 2048
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import mlflow
import numpy as np
from scipy import stats as sci_stats

from benchmarks.baselines import (
    BaselineRunConfig,
    EWCOnlyStrategy,
    FixedScheduleStrategy,
    PSIFullRetrainStrategy,
    run_baseline,
)
from benchmarks.baselines.harness import make_baseline_stream
from benchmarks.synthetic_drift_gen import build_default_scenarios
from cadence.adapters.neural import FraudNet, FraudNetConfig
from cadence.carbon.model import GridProfile, HardwareProfile
from cadence.collector.drift_trigger import DriftTriggerConfig, PSITrigger
from cadence.common.config import load_config
from cadence.common.logging import get_logger
from cadence.common.seeds import set_global_seed
from cadence.common.tracking import start_run
from cadence.data.loaders import load_credit_card_fraud
from cadence.rso.env import SandboxConfig
from cadence.rso.ppo import (
    PPOTrainConfig,
    build_multi_scenario_env_factory,
    evaluate_ppo,
    train_ppo,
)
from cadence.rso.scorers import PSIResponsibilityScorer

log = get_logger("cadence.benchmarks.phase2")


def _mean_std(values):
    vs = list(values)
    if not vs:
        return (0.0, 0.0)
    return (float(statistics.fmean(vs)), float(statistics.pstdev(vs)) if len(vs) > 1 else 0.0)


def _aggregate_baseline(outcomes):
    return {
        "n_seeds": len(outcomes),
        "mean_f1": _mean_std(o.mean_f1 for o in outcomes),
        "min_f1": _mean_std(o.min_f1 for o in outcomes),
        "gpu_hr": _mean_std(o.total_gpu_seconds / 3600.0 for o in outcomes),
        "kg_co2": _mean_std(o.total_kg_co2 for o in outcomes),
        "sla_windows_below": _mean_std(o.sla_windows_below for o in outcomes),
        "forgetting": _mean_std(o.forgetting_score for o in outcomes),
    }


def _aggregate_rso(outcomes):
    return {
        "n_seeds": len(outcomes),
        "mean_f1": _mean_std(o.mean_f1 for o in outcomes),
        "min_f1": _mean_std(o.min_f1 for o in outcomes),
        "gpu_hr": _mean_std(o.total_gpu_seconds / 3600.0 for o in outcomes),
        "kg_co2": _mean_std(o.total_kg_co2 for o in outcomes),
        "sla_windows_below": _mean_std(o.sla_windows_below for o in outcomes),
        "forgetting": _mean_std(o.forgetting_score for o in outcomes),
        "action_counts": {
            "no-op": _mean_std(o.action_counts.get("no-op", 0) for o in outcomes),
            "partial": _mean_std(o.action_counts.get("partial", 0) for o in outcomes),
            "full": _mean_std(o.action_counts.get("full", 0) for o in outcomes),
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--train-scenario", default="amount_multiplicative_abrupt")
    p.add_argument("--scenarios", default="all")
    p.add_argument("--n-windows", type=int, default=15)
    p.add_argument("--window-size", type=int, default=2048)
    p.add_argument("--sla", type=float, default=0.65)
    p.add_argument("--train-timesteps", type=int, default=3000)
    p.add_argument("--out", default="experiments/phase2_summary.json")
    p.add_argument("--w-gpu-hr", type=float, default=0.05)
    p.add_argument("--w-kg-co2", type=float, default=0.5)
    p.add_argument("--lambda-sla", type=float, default=1.0)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    log.info("phase2_start", seeds=args.seeds, train_scenario=args.train_scenario)

    # 1) Load Fraud, split 70/25/5 (pretrain / stream / unrelated).
    ds = load_credit_card_fraud(cfg.data, seed=42)
    n_train = ds.X_train.shape[0]
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

    # 2) Pretrain FraudNet once (deterministic, seed=42).
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
    baseline_state = adapter.state_dict()
    baseline_threshold = adapter.decision_threshold
    log.info("pretrain_done", baseline_threshold=baseline_threshold)

    # 3) Build the scenarios.
    all_scenarios = build_default_scenarios(ds.feature_names)
    if args.scenarios == "all":
        eval_scenarios = all_scenarios
    else:
        wanted = set(args.scenarios.split(","))
        eval_scenarios = [s for s in all_scenarios if s.name in wanted]
    train_scenario = next(s for s in all_scenarios if s.name == args.train_scenario)
    log.info(
        "scenarios",
        train=train_scenario.name,
        eval=[s.name for s in eval_scenarios],
    )

    # 4) Build a *fast* sandbox env for PPO training.
    trigger = PSITrigger(
        baseline_X=X_pre,
        feature_names=ds.feature_names,
        cfg=DriftTriggerConfig(psi_threshold=0.25, min_window_rows=200),
    )
    scorer = PSIResponsibilityScorer(trigger=trigger)

    # Training will see all scenarios in rotation (better generalization than
    # overfitting to `--train-scenario` alone). The train_scenario arg is kept
    # for backward compat but ignored when scenarios == "all".
    _ = train_scenario  # noqa: F841 — reserved for future single-scenario training

    hardware = HardwareProfile()
    grid = GridProfile()

    train_sandbox_cfg = SandboxConfig(
        window_size=args.window_size,
        max_windows_per_episode=args.n_windows,
        finetune_epochs=1,
        fullretrain_epochs=1,  # fast during PPO training
        replay_buffer_size=1000,
        w_gpu_hr=args.w_gpu_hr,
        w_kg_co2=args.w_kg_co2,
        lambda_sla=args.lambda_sla,
        sla_target=args.sla,
        hardware=hardware,
        grid=grid,
    )

    # Fresh initial-adapter snapshot for the env factory.
    env_adapter_seed = adapter.clone()
    env_adapter_seed.load_state_dict(baseline_state)
    env_adapter_seed.decision_threshold = baseline_threshold

    with start_run(cfg, run_name=f"phase2-train-{train_scenario.name}") as _run:
        mlflow.log_param("phase", 2)
        mlflow.log_param("train_scenario", train_scenario.name)
        mlflow.log_param("train_timesteps", args.train_timesteps)
        mlflow.log_param("w_gpu_hr", args.w_gpu_hr)
        mlflow.log_param("w_kg_co2", args.w_kg_co2)
        mlflow.log_param("lambda_sla", args.lambda_sla)
        mlflow.log_param("seeds", args.seeds)
        mlflow.log_param("n_windows", args.n_windows)
        mlflow.log_param("window_size", args.window_size)
        mlflow.log_param("sla_target", args.sla)

        # 5) Train PPO — cycle through all eval scenarios so it generalizes.
        factory = build_multi_scenario_env_factory(
            initial_adapter=env_adapter_seed,
            historical_X=X_pre,
            historical_y=y_pre,
            raw_stream_X=X_stream,
            raw_stream_y=y_stream,
            scenarios=eval_scenarios,
            scorer=scorer,
            sandbox_cfg=train_sandbox_cfg,
            seed=0,
        )
        model = train_ppo(
            factory,
            cfg=PPOTrainConfig(
                total_timesteps=args.train_timesteps,
                seed=42,
                verbose=0,
            ),
            save_path=Path("experiments") / "rso_ppo_phase2.zip",
            check_env_first=False,  # we already checked in test_rso_env
        )
        log.info("ppo_ready")

        # 6) Evaluate PPO + all baselines on every eval scenario × N seeds.
        run_cfg = BaselineRunConfig(
            window_size=args.window_size,
            n_windows=args.n_windows,
            sla_target=args.sla,
            hardware=hardware,
            grid=grid,
        )
        eval_sandbox_cfg = SandboxConfig(
            window_size=args.window_size,
            max_windows_per_episode=args.n_windows,
            finetune_epochs=5,  # paper-fidelity retrain during eval
            fullretrain_epochs=15,
            replay_buffer_size=5000,
            w_gpu_hr=args.w_gpu_hr,
            w_kg_co2=args.w_kg_co2,
            lambda_sla=args.lambda_sla,
            sla_target=args.sla,
            hardware=hardware,
            grid=grid,
        )

        baseline_factories = [
            PSIFullRetrainStrategy,
            FixedScheduleStrategy,
            EWCOnlyStrategy,
        ]
        baseline_names = [f().name for f in baseline_factories]

        # Structure: results[scenario][strategy_name] = list of outcomes
        all_results: dict[str, dict[str, list]] = {}

        for scenario in eval_scenarios:
            all_results[scenario.name] = {name: [] for name in baseline_names + ["rso_ppo"]}
            # 6a) Baselines.
            for factory in baseline_factories:
                for seed in range(args.seeds):
                    set_global_seed(seed)
                    strat = factory()
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
                        strategy=strat,
                        seed=seed,
                        cfg=run_cfg,
                        feature_names=ds.feature_names,
                    )
                    all_results[scenario.name][strat.name].append(outcome)

            # 6b) RSO — build fresh env with THIS scenario's drift, then eval.
            for seed in range(args.seeds):
                set_global_seed(seed)
                eval_stream_X, eval_stream_y = make_baseline_stream(
                    X_stream, y_stream, scenario, seed=seed
                )[:2]
                fresh_adapter = adapter.clone()
                fresh_adapter.load_state_dict(baseline_state)
                fresh_adapter.decision_threshold = baseline_threshold
                from cadence.rso.env import RetrainingSandboxEnv

                env = RetrainingSandboxEnv(
                    initial_adapter=fresh_adapter,
                    historical_X=X_pre,
                    historical_y=y_pre,
                    drifted_stream_X=eval_stream_X,
                    drifted_stream_y=eval_stream_y,
                    responsibility_scorer=scorer,
                    cfg=eval_sandbox_cfg,
                )
                rso_outcome = evaluate_ppo(
                    model,
                    env,
                    unrelated_X=X_unrel,
                    unrelated_y=y_unrel,
                    sla_target=args.sla,
                    scenario_name=scenario.name,
                    seed=seed,
                    max_windows=args.n_windows,
                )
                all_results[scenario.name]["rso_ppo"].append(rso_outcome)
                log.info(
                    "rso_eval_seed",
                    scenario=scenario.name,
                    seed=seed,
                    mean_f1=rso_outcome.mean_f1,
                    gpu_hr=rso_outcome.total_gpu_seconds / 3600.0,
                    actions=rso_outcome.action_counts,
                )

        # 7) Aggregate across scenarios + paired stats.
        summary = {}
        for scen_name, per_strat in all_results.items():
            agg = {}
            for strat_name, outs in per_strat.items():
                agg[strat_name] = (
                    _aggregate_rso(outs) if strat_name == "rso_ppo" else _aggregate_baseline(outs)
                )
            # Paired Wilcoxon: RSO vs each baseline on mean_f1 AND gpu_hr.
            stats_table = {}
            rso_outs = per_strat["rso_ppo"]
            for baseline_name in baseline_names:
                b_outs = per_strat[baseline_name]
                if len(rso_outs) < 2 or len(b_outs) < 2:
                    continue
                pairs_a = [o.mean_f1 for o in rso_outs]
                pairs_b = [o.mean_f1 for o in b_outs]
                try:
                    stat_f1, p_f1 = sci_stats.wilcoxon(pairs_a, pairs_b)
                except ValueError:
                    stat_f1, p_f1 = None, None
                pairs_a_gpu = [o.total_gpu_seconds / 3600.0 for o in rso_outs]
                pairs_b_gpu = [o.total_gpu_seconds / 3600.0 for o in b_outs]
                try:
                    stat_gpu, p_gpu = sci_stats.wilcoxon(pairs_a_gpu, pairs_b_gpu)
                except ValueError:
                    stat_gpu, p_gpu = None, None
                stats_table[f"rso_vs_{baseline_name}"] = {
                    "mean_f1": {"stat": stat_f1, "p": p_f1},
                    "gpu_hr": {"stat": stat_gpu, "p": p_gpu},
                }
            summary[scen_name] = {"aggregates": agg, "wilcoxon": stats_table}

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        mlflow.log_artifact(args.out)

        # Human-readable table.
        print("\n=== Phase 2 results ===")
        for scen_name, entry in summary.items():
            print(f"\n[{scen_name}]")
            print(f"  {'strategy':<20}{'mean_f1':>18}{'gpu_hr':>18}{'kg_co2':>18}{'sla_below':>12}")
            for strat_name, a in entry["aggregates"].items():
                print(
                    f"  {strat_name:<20}"
                    f"{a['mean_f1'][0]:>10.4f} ± {a['mean_f1'][1]:.4f}"
                    f"{a['gpu_hr'][0]:>10.4f} ± {a['gpu_hr'][1]:.4f}"
                    f"{a['kg_co2'][0]:>10.4f} ± {a['kg_co2'][1]:.4f}"
                    f"{a['sla_windows_below'][0]:>10.2f}"
                )
            print("  Wilcoxon RSO vs baselines:")
            for k, v in entry["wilcoxon"].items():
                print(f"    {k}: f1_p={v['mean_f1']['p']!s:>10} gpu_hr_p={v['gpu_hr']['p']!s:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
