"""Phase F, part 2: real H3 on Split-MNIST — the honest forgetting test.

Step D's Gate D found H3 was NOT supported on Credit Card Fraud because
Fraud is single-task. This runner replaces Fraud with Split-MNIST, where
Task 0 ({0,1}) and Task 1 ({2,3}) are *genuinely* disjoint classes.

Protocol per seed:
  1. Pretrain FraudNet-style MLP on Task 0.
  2. Snapshot pre_task0_F1 on Task-0 held-out test set.
  3. Retrain on Task 1 training data — two strategies:
     * PARTIAL: EWC-regularized fine-tune of `layer1` (Fisher weighted
       from Task-0 data; Part-3B mechanics).
     * FULL:    naive full retrain on Task 1 (no EWC, no replay).
  4. Measure post_task0_F1 after each retrain.
  5. forgetting = pre_task0_F1 - post_task0_F1 (higher = worse).
  6. Paired Wilcoxon partial < full — H3 supported iff p < 0.05.

Usage:
    python -m benchmarks.phase_f_mnist --seeds 5
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
from sklearn.metrics import f1_score

from cadence.adapters.neural import FraudNet, FraudNetConfig
from cadence.common.config import load_config
from cadence.common.device import log_device_info
from cadence.common.logging import get_logger
from cadence.common.seeds import set_global_seed
from cadence.common.tracking import start_run
from cadence.data.mnist_splits import load_split_mnist
from cadence.executor import ExecutorConfig, RetrainExecutor

log = get_logger("cadence.benchmarks.phase_f_mnist")


def _f1(adapter, X, y) -> float:
    if X.size == 0:
        return 0.0
    probs = adapter.predict_proba(X)
    preds = (probs >= adapter.decision_threshold).astype(np.int64)
    return float(f1_score(y, preds, zero_division=0))


def _mean_std(vs):
    vs = list(vs)
    if not vs:
        return (0.0, 0.0)
    return (
        float(statistics.fmean(vs)),
        float(statistics.pstdev(vs)) if len(vs) > 1 else 0.0,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--task-a", type=int, nargs=2, default=[0, 1])
    p.add_argument("--task-b", type=int, nargs=2, default=[2, 3])
    p.add_argument("--finetune-epochs", type=int, default=5)
    p.add_argument("--fullretrain-epochs", type=int, default=8)
    p.add_argument("--sla", type=float, default=0.90)
    p.add_argument("--target-layer", default="layer1")
    p.add_argument("--ewc-penalty", type=float, default=5000.0)
    p.add_argument("--out", default="experiments/phase_f_mnist.json")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    log_device_info()
    log.info(
        "phase_f_mnist_start",
        seeds=args.seeds,
        task_a=args.task_a,
        task_b=args.task_b,
        ewc_penalty=args.ewc_penalty,
    )

    # Load train + test splits, keep only tasks A and B.
    train_tasks = load_split_mnist(split="train", tasks=(tuple(args.task_a), tuple(args.task_b)))
    test_tasks = load_split_mnist(split="test", tasks=(tuple(args.task_a), tuple(args.task_b)))
    task_a_train, task_b_train = train_tasks
    task_a_test, task_b_test = test_tasks
    log.info(
        "tasks_loaded",
        task_a_n=int(task_a_train.X.shape[0]),
        task_b_n=int(task_b_train.X.shape[0]),
        task_a_test_n=int(task_a_test.X.shape[0]),
        task_b_test_n=int(task_b_test.X.shape[0]),
    )

    def _new_adapter() -> FraudNet:
        return FraudNet(
            FraudNetConfig(
                input_dim=784,
                hidden_dims=list(cfg.model.hidden_dims),
                dropout=cfg.model.dropout,
                lr=cfg.model.lr,
                batch_size=128,
                max_epochs=cfg.model.max_epochs,
                early_stopping_patience=cfg.model.early_stopping_patience,
                class_weighted=cfg.model.class_weighted,
            )
        )

    forgetting_partial: list[float] = []
    forgetting_full: list[float] = []
    task_b_partial_f1: list[float] = []
    task_b_full_f1: list[float] = []
    pre_task_a_f1_all: list[float] = []

    with start_run(cfg, run_name=f"phase_f_mnist_ta{args.task_a}_tb{args.task_b}"):
        mlflow.log_param("phase", "F/mnist")
        mlflow.log_param("task_a", str(args.task_a))
        mlflow.log_param("task_b", str(args.task_b))
        mlflow.log_param("seeds", args.seeds)
        mlflow.log_param("ewc_penalty", args.ewc_penalty)
        mlflow.log_param("finetune_epochs", args.finetune_epochs)

        for seed in range(args.seeds):
            set_global_seed(seed)

            # Pretrain on Task A.
            adapter = _new_adapter()
            adapter.fit(
                task_a_train.X,
                task_a_train.y,
                X_val=task_a_test.X[:1500],
                y_val=task_a_test.y[:1500],
            )
            baseline_state = adapter.state_dict()
            baseline_threshold = adapter.decision_threshold
            pre_task_a_f1 = _f1(adapter, task_a_test.X, task_a_test.y)
            pre_task_a_f1_all.append(pre_task_a_f1)

            # ---------- Path A: PARTIAL EWC on Task B ----------
            adapter_p = adapter.clone()
            adapter_p.load_state_dict(baseline_state)
            adapter_p.decision_threshold = baseline_threshold
            exec_p = RetrainExecutor(
                adapter_p,
                historical_X=task_a_train.X,
                historical_y=task_a_train.y,
                cfg=ExecutorConfig(
                    sla_target=args.sla,
                    finetune_epochs=args.finetune_epochs,
                    fullretrain_epochs=args.fullretrain_epochs,
                    ewc_penalty=args.ewc_penalty,
                    ewc_fisher_sample_size=2000,
                    # We don't want the executor to rollback (that would defeat
                    # the H3 measurement) — set the promotion threshold to 0.
                    promotion_threshold_ratio=0.0,
                ),
            )
            exec_p.execute(
                action="partial",
                window_X=task_b_train.X,
                window_y=task_b_train.y,
                target_layer=args.target_layer,
            )
            post_partial_task_a = _f1(adapter_p, task_a_test.X, task_a_test.y)
            post_partial_task_b = _f1(adapter_p, task_b_test.X, task_b_test.y)
            forget_p = pre_task_a_f1 - post_partial_task_a

            # ---------- Path B: FULL naive on Task B ----------
            adapter_f = adapter.clone()
            adapter_f.load_state_dict(baseline_state)
            adapter_f.decision_threshold = baseline_threshold
            exec_f = RetrainExecutor(
                adapter_f,
                historical_X=task_a_train.X,
                historical_y=task_a_train.y,
                cfg=ExecutorConfig(
                    sla_target=args.sla,
                    finetune_epochs=args.finetune_epochs,
                    fullretrain_epochs=args.fullretrain_epochs,
                    promotion_threshold_ratio=0.0,
                ),
            )
            # Full retrain here is *only* on Task B (no replay of Task A) —
            # this is the strict "naive full retrain that ignores prior task"
            # baseline the H3 test needs to isolate the value of EWC. If we
            # let it see Task A via `historical_X`, we'd be conflating
            # replay-buffer effects with EWC. Bypass the executor's replay
            # concatenation by calling the adapter directly.
            adapter_f.fit(task_b_train.X, task_b_train.y)
            post_full_task_a = _f1(adapter_f, task_a_test.X, task_a_test.y)
            post_full_task_b = _f1(adapter_f, task_b_test.X, task_b_test.y)
            forget_f = pre_task_a_f1 - post_full_task_a

            forgetting_partial.append(forget_p)
            forgetting_full.append(forget_f)
            task_b_partial_f1.append(post_partial_task_b)
            task_b_full_f1.append(post_full_task_b)
            log.info(
                "seed_done",
                seed=seed,
                pre_task_a_f1=pre_task_a_f1,
                post_partial_task_a=post_partial_task_a,
                post_full_task_a=post_full_task_a,
                post_partial_task_b=post_partial_task_b,
                post_full_task_b=post_full_task_b,
                forget_partial=forget_p,
                forget_full=forget_f,
            )
            mlflow.log_metric(f"pre_task_a_f1_seed{seed}", pre_task_a_f1)
            mlflow.log_metric(f"forget_partial_seed{seed}", forget_p)
            mlflow.log_metric(f"forget_full_seed{seed}", forget_f)

            del adapter, adapter_p, adapter_f, exec_p, exec_f
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
            "task_a": list(args.task_a),
            "task_b": list(args.task_b),
            "seeds": args.seeds,
            "ewc_penalty": args.ewc_penalty,
            "pre_task_a_f1": _mean_std(pre_task_a_f1_all),
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
            "task_b_partial_f1": _mean_std(task_b_partial_f1),
            "task_b_full_f1": _mean_std(task_b_full_f1),
            "wilcoxon_partial_less_full": wilcoxon,
            "config": vars(args),
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        mlflow.log_artifact(args.out)
        if wilcoxon.get("p") is not None:
            mlflow.log_metric("wilcoxon_p_partial_less_full", wilcoxon["p"])
        mlflow.log_metric("forgetting_partial_mean", summary["forgetting_partial"]["mean"])
        mlflow.log_metric("forgetting_full_mean", summary["forgetting_full"]["mean"])

        print(f"\n=== Gate F (MNIST H3): Task A={args.task_a} -> Task B={args.task_b} ===")
        print(f"Pre Task A F1: {summary['pre_task_a_f1'][0]:.4f} +/- {summary['pre_task_a_f1'][1]:.4f}")
        print(
            f"Forgetting (partial): {summary['forgetting_partial']['mean']:+.4f} "
            f"+/- {summary['forgetting_partial']['std']:.4f}"
        )
        print(
            f"Forgetting (full)   : {summary['forgetting_full']['mean']:+.4f} "
            f"+/- {summary['forgetting_full']['std']:.4f}"
        )
        print(
            f"Task B F1 (partial) : {summary['task_b_partial_f1'][0]:.4f} +/- {summary['task_b_partial_f1'][1]:.4f}"
        )
        print(
            f"Task B F1 (full)    : {summary['task_b_full_f1'][0]:.4f} +/- {summary['task_b_full_f1'][1]:.4f}"
        )
        print(
            f"Wilcoxon partial<full: stat={wilcoxon.get('stat')} "
            f"p={wilcoxon.get('p')}  n_pairs={wilcoxon.get('n_pairs')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
