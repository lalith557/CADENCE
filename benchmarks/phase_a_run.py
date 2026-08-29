"""Gate A runner — Step A of the Execution Plan.

Anchors Claim C on the contested-SLA scenarios by:
  1. Adding 3 contested-SLA scenarios where "no-op" is provably wrong on
     every seed (post-drift F1 is calibrated to sit below the SLA target).
  2. Wiring EWC into the sandbox's partial-retrain path so the sandbox
     matches Part 3B mechanics (fixes W-23).
  3. Wrapping the training env in AugmentedLagrangianEnv so λ_sla adapts
     to the actual violation rate rather than being pinned at a
     hand-picked constant (fixes W-22).
  4. Training PPO for `--train-timesteps` (default 30_000, up from R-4's
     3_000) so the policy has room to escape bang-bang.
  5. Evaluating RSO + all Phase 1 baselines on the full Step A scenario
     set with ≥10 seeds and paired Wilcoxon per scenario.

If RSO beats periodic-retrain and reactive-full-retrain on GPU-hr at
equal-or-better F1 (Wilcoxon p<0.05), Gate A passes and R-Gate-A goes
into docs/results.md. If not, we record the negative honestly.

Usage:
    python -m benchmarks.phase_a_run --config configs/default.yaml \
        --seeds 10 --train-timesteps 30000 --sla 0.65 \
        --n-windows 15 --window-size 2048
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
from benchmarks.synthetic_drift_gen import (
    build_contested_sla_scenarios,
    build_step_a_scenarios,
)
from cadence.adapters.neural import FraudNet, FraudNetConfig
from cadence.carbon.model import GridProfile, HardwareProfile
from cadence.collector.drift_trigger import DriftTriggerConfig, PSITrigger
from cadence.common.config import load_config
from cadence.common.logging import get_logger
from cadence.common.seeds import set_global_seed
from cadence.common.tracking import start_run
from cadence.data.loaders import load_credit_card_fraud
from cadence.rso.env import RetrainingSandboxEnv, SandboxConfig
from cadence.rso.lagrangian import AugmentedLagrangianConfig, AugmentedLagrangianEnv
from cadence.rso.ppo import (
    PPOTrainConfig,
    build_multi_scenario_env_factory,
    evaluate_ppo,
    train_ppo,
)
from cadence.rso.scorers import PSIResponsibilityScorer

log = get_logger("cadence.benchmarks.phase_a")


def _mean_std(values):
    vs = list(values)
    if not vs:
        return (0.0, 0.0)
    return (
        float(statistics.fmean(vs)),
        float(statistics.pstdev(vs)) if len(vs) > 1 else 0.0,
    )


def _aggregate(outcomes):
    return {
        "n_seeds": len(outcomes),
        "mean_f1": _mean_std(o.mean_f1 for o in outcomes),
        "min_f1": _mean_std(o.min_f1 for o in outcomes),
        "gpu_hr": _mean_std(o.total_gpu_seconds / 3600.0 for o in outcomes),
        "kg_co2": _mean_std(o.total_kg_co2 for o in outcomes),
        "sla_windows_below": _mean_std(o.sla_windows_below for o in outcomes),
        "forgetting": _mean_std(o.forgetting_score for o in outcomes),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--n-windows", type=int, default=15)
    p.add_argument("--window-size", type=int, default=2048)
    p.add_argument("--sla", type=float, default=0.65)
    p.add_argument("--train-timesteps", type=int, default=30_000)
    p.add_argument("--w-gpu-hr", type=float, default=0.05)
    p.add_argument("--w-kg-co2", type=float, default=0.5)
    p.add_argument("--lambda-init", type=float, default=1.0)
    p.add_argument("--dual-lr", type=float, default=5.0)
    p.add_argument("--ewc-penalty", type=float, default=1000.0)
    p.add_argument("--contested-only", action="store_true",
                   help="Restrict eval to just the 3 contested-SLA scenarios.")
    p.add_argument("--out", default="experiments/phase_a_summary.json")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    log.info("phase_a_start", seeds=args.seeds, train_timesteps=args.train_timesteps)

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
    X_pre, y_pre = ds.X_train[pretrain_idx], ds.y_train[pretrain_idx]
    X_stream, y_stream = ds.X_train[stream_idx], ds.y_train[stream_idx]
    X_unrel, y_unrel = ds.X_train[unrelated_idx], ds.y_train[unrelated_idx]

    # 2) Pretrain FraudNet once (deterministic, seed=42) on GPU.
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

    # 3) Scenarios.
    if args.contested_only:
        eval_scenarios = build_contested_sla_scenarios(ds.feature_names)
    else:
        eval_scenarios = build_step_a_scenarios(ds.feature_names)
    log.info("scenarios", eval=[s.name for s in eval_scenarios])

    trigger = PSITrigger(
        baseline_X=X_pre,
        feature_names=ds.feature_names,
        cfg=DriftTriggerConfig(psi_threshold=0.25, min_window_rows=200),
    )
    scorer = PSIResponsibilityScorer(trigger=trigger)
    hardware = HardwareProfile()
    grid = GridProfile()

    # 4) Fast sandbox for PPO training + Augmented-Lagrangian wrapping.
    train_sandbox_cfg = SandboxConfig(
        window_size=args.window_size,
        max_windows_per_episode=args.n_windows,
        finetune_epochs=1,
        fullretrain_epochs=1,
        replay_buffer_size=1000,
        w_gpu_hr=args.w_gpu_hr,
        w_kg_co2=args.w_kg_co2,
        lambda_sla=args.lambda_init,
        sla_target=args.sla,
        ewc_penalty=args.ewc_penalty,
        ewc_fisher_sample_size=1000,
        ewc_cache_fisher=True,
        hardware=hardware,
        grid=grid,
    )

    env_adapter_seed = adapter.clone()
    env_adapter_seed.load_state_dict(baseline_state)
    env_adapter_seed.decision_threshold = baseline_threshold

    dual_cfg = AugmentedLagrangianConfig(
        lambda_init=args.lambda_init,
        dual_lr=args.dual_lr,
    )

    with start_run(cfg, run_name="phase_a") as _run:
        mlflow.log_param("phase", "A")
        mlflow.log_param("train_timesteps", args.train_timesteps)
        mlflow.log_param("w_gpu_hr", args.w_gpu_hr)
        mlflow.log_param("w_kg_co2", args.w_kg_co2)
        mlflow.log_param("lambda_init", args.lambda_init)
        mlflow.log_param("dual_lr", args.dual_lr)
        mlflow.log_param("ewc_penalty", args.ewc_penalty)
        mlflow.log_param("seeds", args.seeds)
        mlflow.log_param("n_windows", args.n_windows)
        mlflow.log_param("window_size", args.window_size)
        mlflow.log_param("sla_target", args.sla)
        mlflow.log_param("n_scenarios", len(eval_scenarios))

        base_factory = build_multi_scenario_env_factory(
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

        def _wrapped_factory():
            env = base_factory()
            return AugmentedLagrangianEnv(env, cfg=dual_cfg)

        model = train_ppo(
            _wrapped_factory,
            cfg=PPOTrainConfig(
                total_timesteps=args.train_timesteps,
                seed=42,
                verbose=0,
            ),
            save_path=Path("experiments") / "rso_ppo_phase_a.zip",
            check_env_first=False,
        )
        log.info("ppo_ready")

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
            finetune_epochs=5,
            fullretrain_epochs=15,
            replay_buffer_size=5000,
            w_gpu_hr=args.w_gpu_hr,
            w_kg_co2=args.w_kg_co2,
            lambda_sla=args.lambda_init,
            sla_target=args.sla,
            ewc_penalty=args.ewc_penalty,
            ewc_fisher_sample_size=2000,
            ewc_cache_fisher=True,
            hardware=hardware,
            grid=grid,
        )

        baseline_factories = [
            PSIFullRetrainStrategy,
            FixedScheduleStrategy,
            EWCOnlyStrategy,
        ]
        baseline_names = [f().name for f in baseline_factories]

        all_results: dict[str, dict[str, list]] = {}

        for scenario in eval_scenarios:
            all_results[scenario.name] = {name: [] for name in baseline_names + ["rso_ppo"]}
            log.info("eval_scenario_start", name=scenario.name)

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

            for seed in range(args.seeds):
                set_global_seed(seed)
                stream_pair = make_baseline_stream(X_stream, y_stream, scenario, seed=seed)
                eval_stream_X, eval_stream_y = stream_pair[0], stream_pair[1]
                fresh_adapter = adapter.clone()
                fresh_adapter.load_state_dict(baseline_state)
                fresh_adapter.decision_threshold = baseline_threshold
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

        # Aggregate + paired stats.
        summary = {"scenarios": {}, "config": vars(args)}
        for scen_name, per_strat in all_results.items():
            agg = {name: _aggregate(outs) for name, outs in per_strat.items()}
            stats_table = {}
            rso_outs = per_strat["rso_ppo"]
            for baseline_name in baseline_names:
                b_outs = per_strat[baseline_name]
                if len(rso_outs) < 2 or len(b_outs) < 2:
                    continue
                try:
                    _, p_f1 = sci_stats.wilcoxon(
                        [o.mean_f1 for o in rso_outs],
                        [o.mean_f1 for o in b_outs],
                    )
                except ValueError:
                    p_f1 = None
                try:
                    _, p_gpu = sci_stats.wilcoxon(
                        [o.total_gpu_seconds / 3600.0 for o in rso_outs],
                        [o.total_gpu_seconds / 3600.0 for o in b_outs],
                    )
                except ValueError:
                    p_gpu = None
                stats_table[f"rso_vs_{baseline_name}"] = {
                    "p_mean_f1": None if p_f1 is None else float(p_f1),
                    "p_gpu_hr": None if p_gpu is None else float(p_gpu),
                }
            summary["scenarios"][scen_name] = {"aggregates": agg, "wilcoxon": stats_table}

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        mlflow.log_artifact(args.out)

        print("\n=== Gate A results ===")
        for scen_name, entry in summary["scenarios"].items():
            print(f"\n[{scen_name}]")
            print(
                f"  {'strategy':<20}{'mean_f1':>18}{'gpu_hr':>18}{'kg_co2':>18}{'sla_below':>12}"
            )
            for strat_name, a in entry["aggregates"].items():
                print(
                    f"  {strat_name:<20}"
                    f"{a['mean_f1'][0]:>10.4f} +/- {a['mean_f1'][1]:.4f}"
                    f"{a['gpu_hr'][0]:>10.4f} +/- {a['gpu_hr'][1]:.4f}"
                    f"{a['kg_co2'][0]:>10.4f} +/- {a['kg_co2'][1]:.4f}"
                    f"{a['sla_windows_below'][0]:>10.2f}"
                )
            print("  Wilcoxon RSO vs baselines:")
            for k, v in entry["wilcoxon"].items():
                print(f"    {k}: p_f1={v['p_mean_f1']} p_gpu={v['p_gpu_hr']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
