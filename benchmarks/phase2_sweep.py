"""Reward-weight sensitivity sweep (W-7).

For each (w_gpu_hr, lambda_sla) combination in a small grid, retrain PPO and
evaluate on 3 seeds × 2 representative scenarios. Report mean_f1 and gpu_hr
per cell. The "working range" is the region where mean_f1 stays close to the
paper baseline while gpu_hr shrinks.

This is a MINIMAL sweep — 3×3 = 9 cells — chosen so the whole thing runs in
under an hour. A wider sweep (5×5) is future work.

Explicitly calls gc.collect() + torch.cuda.empty_cache() between cells so
retained PPO / env / adapter state doesn't accumulate over the sweep (an
earlier attempt with no GC exited mid-run with no traceback — likely an OOM).

Usage:
    python -m benchmarks.phase2_sweep --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.baselines.harness import make_baseline_stream
from benchmarks.synthetic_drift_gen import build_default_scenarios
from cadence.adapters.neural import FraudNet, FraudNetConfig
from cadence.carbon.model import GridProfile, HardwareProfile
from cadence.collector.drift_trigger import DriftTriggerConfig, PSITrigger
from cadence.common.config import load_config
from cadence.common.logging import get_logger
from cadence.common.seeds import set_global_seed
from cadence.data.loaders import load_credit_card_fraud
from cadence.rso.env import RetrainingSandboxEnv, SandboxConfig
from cadence.rso.ppo import (
    PPOTrainConfig,
    build_multi_scenario_env_factory,
    evaluate_ppo,
    train_ppo,
)
from cadence.rso.scorers import PSIResponsibilityScorer

log = get_logger("cadence.benchmarks.phase2_sweep")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-windows", type=int, default=15)
    p.add_argument("--window-size", type=int, default=2048)
    p.add_argument("--sla", type=float, default=0.65)
    p.add_argument("--train-timesteps", type=int, default=1500)
    p.add_argument("--out", default="experiments/phase2_sweep.json")
    args = p.parse_args(argv)

    cfg = load_config(args.config)

    # Setup identical to phase2_run.
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
    X_unrel = ds.X_train[idx[n_pretrain + n_stream :]]
    y_unrel = ds.y_train[idx[n_pretrain + n_stream :]]

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

    scenarios = build_default_scenarios(ds.feature_names)
    # Two scenarios: one abrupt covariate, one gradual — cover both regimes.
    eval_names = ["amount_multiplicative_abrupt", "amount_multiplicative_gradual"]
    eval_scenarios = [s for s in scenarios if s.name in eval_names]

    trigger = PSITrigger(
        baseline_X=X_pre,
        feature_names=ds.feature_names,
        cfg=DriftTriggerConfig(psi_threshold=0.25, min_window_rows=200),
    )
    scorer = PSIResponsibilityScorer(trigger=trigger)

    # Sweep grid (small).
    w_gpu_grid = [0.01, 0.05, 0.20]
    lambda_grid = [0.5, 1.0, 2.0]

    hardware = HardwareProfile()
    grid_profile = GridProfile()

    results: dict = {"grid": {"w_gpu_hr": w_gpu_grid, "lambda_sla": lambda_grid}, "cells": []}

    env_adapter_seed = adapter.clone()
    env_adapter_seed.load_state_dict(baseline_state)
    env_adapter_seed.decision_threshold = baseline_threshold

    for w_gpu in w_gpu_grid:
        for lam in lambda_grid:
            log.info("sweep_cell_start", w_gpu_hr=w_gpu, lambda_sla=lam)
            train_cfg = SandboxConfig(
                window_size=args.window_size,
                max_windows_per_episode=args.n_windows,
                finetune_epochs=1,
                fullretrain_epochs=1,
                replay_buffer_size=1000,
                w_gpu_hr=w_gpu,
                w_kg_co2=0.5,
                lambda_sla=lam,
                sla_target=args.sla,
                hardware=hardware,
                grid=grid_profile,
            )
            factory = build_multi_scenario_env_factory(
                initial_adapter=env_adapter_seed,
                historical_X=X_pre,
                historical_y=y_pre,
                raw_stream_X=X_stream,
                raw_stream_y=y_stream,
                scenarios=eval_scenarios,
                scorer=scorer,
                sandbox_cfg=train_cfg,
                seed=0,
            )
            model = train_ppo(
                factory,
                cfg=PPOTrainConfig(
                    total_timesteps=args.train_timesteps,
                    seed=42,
                    verbose=0,
                ),
                save_path=None,
                check_env_first=False,
            )

            # Evaluate on each scenario × seeds.
            eval_cfg = SandboxConfig(
                window_size=args.window_size,
                max_windows_per_episode=args.n_windows,
                finetune_epochs=5,
                fullretrain_epochs=15,
                replay_buffer_size=5000,
                w_gpu_hr=w_gpu,
                w_kg_co2=0.5,
                lambda_sla=lam,
                sla_target=args.sla,
                hardware=hardware,
                grid=grid_profile,
            )
            cell_outcomes = []
            for scenario in eval_scenarios:
                for seed in range(args.seeds):
                    set_global_seed(seed)
                    stream_X, stream_y, _ = make_baseline_stream(
                        X_stream, y_stream, scenario, seed=seed
                    )
                    fresh = adapter.clone()
                    fresh.load_state_dict(baseline_state)
                    fresh.decision_threshold = baseline_threshold
                    env = RetrainingSandboxEnv(
                        initial_adapter=fresh,
                        historical_X=X_pre,
                        historical_y=y_pre,
                        drifted_stream_X=stream_X,
                        drifted_stream_y=stream_y,
                        responsibility_scorer=scorer,
                        cfg=eval_cfg,
                    )
                    outcome = evaluate_ppo(
                        model,
                        env,
                        unrelated_X=X_unrel,
                        unrelated_y=y_unrel,
                        sla_target=args.sla,
                        scenario_name=scenario.name,
                        seed=seed,
                        max_windows=args.n_windows,
                    )
                    cell_outcomes.append(outcome)

            mean_f1 = float(np.mean([o.mean_f1 for o in cell_outcomes]))
            std_f1 = float(np.std([o.mean_f1 for o in cell_outcomes]))
            mean_gpu = float(np.mean([o.total_gpu_seconds / 3600.0 for o in cell_outcomes]))
            std_gpu = float(np.std([o.total_gpu_seconds / 3600.0 for o in cell_outcomes]))
            mean_actions = {
                a: float(np.mean([o.action_counts.get(a, 0) for o in cell_outcomes]))
                for a in ("no-op", "partial", "full")
            }
            cell = {
                "w_gpu_hr": w_gpu,
                "lambda_sla": lam,
                "mean_f1": (mean_f1, std_f1),
                "gpu_hr": (mean_gpu, std_gpu),
                "action_counts": mean_actions,
            }
            results["cells"].append(cell)
            log.info("sweep_cell_done", **cell)

            # Free per-cell state so memory doesn't accumulate across cells.
            del model, factory, cell_outcomes
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Checkpoint the JSON after each cell so a mid-run crash still
            # leaves usable partial results.
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== Phase 2 sensitivity sweep ===")
    print(f"{'w_gpu_hr':>10}{'lambda':>10}{'mean_f1':>18}{'gpu_hr':>18}{'  actions':>28}")
    for c in results["cells"]:
        print(
            f"{c['w_gpu_hr']:>10.3f}{c['lambda_sla']:>10.3f}"
            f"{c['mean_f1'][0]:>10.4f} ± {c['mean_f1'][1]:.4f}"
            f"{c['gpu_hr'][0]:>10.5f} ± {c['gpu_hr'][1]:.5f}"
            f"  no={c['action_counts']['no-op']:.1f} p={c['action_counts']['partial']:.1f} f={c['action_counts']['full']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
