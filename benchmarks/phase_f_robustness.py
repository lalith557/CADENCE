"""Phase F, part 3: robustness — PSI false-alarm rate + noisy-telemetry AUROC.

Two cheap add-on measurements the plan lists under "generality + robustness":

  * **False-alarm rate** — run the PSI trigger over a stream that has NO
    injected drift. Count how often it fires. Compared against the same
    trigger on a drifted stream to give a base-rate control.

  * **Noisy-telemetry robustness** — for the GNN attribution scorer, drop
    a random fraction p ∈ {0.0, 0.1, 0.3, 0.5} of the CDAG's edges (a
    proxy for missing activation samples) and re-score the same drift
    scenario. Report AUROC(root vs rest) as a function of p.

Both write MLflow metrics + a JSON summary artifact.

Usage:
    python -m benchmarks.phase_f_robustness --seeds 5
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
from sklearn.metrics import roc_auc_score

from benchmarks.baselines.harness import make_baseline_stream
from benchmarks.synthetic_drift_gen import build_default_scenarios
from cadence.adapters.neural import FraudNet, FraudNetConfig
from cadence.attribution import (
    GNNConfig,
    GNNResponsibilityScorer,
    GNNTrainConfig,
    generate_sandbox_dataset,
    train_gnn,
)
from cadence.attribution.gnn import cdag_to_pyg_data
from cadence.cdag import build_cdag_from_windows, compute_windowed_signals
from cadence.collector.drift_trigger import DriftTriggerConfig, PSITrigger
from cadence.common.config import load_config
from cadence.common.device import get_device, log_device_info
from cadence.common.logging import get_logger
from cadence.common.seeds import set_global_seed
from cadence.common.tracking import start_run
from cadence.data.loaders import load_credit_card_fraud

log = get_logger("cadence.benchmarks.phase_f_robustness")


def _mean_std(vs):
    vs = list(vs)
    if not vs:
        return (0.0, 0.0)
    return (
        float(statistics.fmean(vs)),
        float(statistics.pstdev(vs)) if len(vs) > 1 else 0.0,
    )


def _drop_edges(weights: np.ndarray, drop_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Randomly zero-out `drop_frac` of the non-zero entries — simulates
    missing activation samples that under-connect the CDAG."""
    if drop_frac <= 0:
        return weights.copy()
    W = weights.copy()
    mask = np.abs(W) > 0
    non_zero_idx = np.argwhere(mask)
    if non_zero_idx.size == 0:
        return W
    n_drop = int(drop_frac * non_zero_idx.shape[0])
    if n_drop == 0:
        return W
    pick = rng.choice(non_zero_idx.shape[0], size=n_drop, replace=False)
    for i in pick:
        r, c = non_zero_idx[i]
        W[r, c] = 0.0
    return W


def _false_alarm_rate(trigger: PSITrigger, stream_X: np.ndarray, *, window_size: int, n_windows: int) -> tuple[int, int]:
    fired = 0
    total = 0
    for w in range(n_windows):
        s = w * window_size
        e = min(s + window_size, stream_X.shape[0])
        if e <= s:
            break
        alert = trigger.evaluate(stream_X[s:e])
        total += 1
        if alert.fired:
            fired += 1
    return fired, total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--window-size", type=int, default=1024)
    p.add_argument("--n-windows", type=int, default=15)
    p.add_argument("--drop-fracs", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5])
    p.add_argument("--gnn-samples", type=int, default=80)
    p.add_argument("--gnn-epochs", type=int, default=12)
    p.add_argument("--out", default="experiments/phase_f_robustness.json")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    dev = get_device()
    log_device_info()

    ds = load_credit_card_fraud(cfg.data, seed=42)
    n_train = ds.X_train.shape[0]
    rng = np.random.default_rng(42)
    idx = np.arange(n_train)
    rng.shuffle(idx)
    n_pre = int(0.70 * n_train)
    X_pre = ds.X_train[idx[:n_pre]]
    y_pre = ds.y_train[idx[:n_pre]]
    X_stream = ds.X_train[idx[n_pre:]]
    y_stream = ds.y_train[idx[n_pre:]]

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

    with start_run(cfg, run_name="phase_f_robustness"):
        mlflow.log_param("phase", "F/robustness")
        mlflow.log_param("seeds", args.seeds)
        mlflow.log_param("window_size", args.window_size)
        mlflow.log_param("n_windows", args.n_windows)
        mlflow.log_param("drop_fracs", str(args.drop_fracs))

        # ---------------- False-alarm rate ----------------
        # On the UNDRIFTED stream (X_stream as-is), how often does PSI fire?
        undrifted_rates: list[float] = []
        drifted_rates: list[float] = []
        for seed in range(args.seeds):
            set_global_seed(seed)
            trigger_un = PSITrigger(
                baseline_X=X_pre,
                feature_names=ds.feature_names,
                cfg=DriftTriggerConfig(psi_threshold=0.25, min_window_rows=200),
            )
            fired_un, total_un = _false_alarm_rate(
                trigger_un, X_stream, window_size=args.window_size, n_windows=args.n_windows
            )
            fa_rate = fired_un / max(total_un, 1)
            undrifted_rates.append(fa_rate)
            log.info("false_alarm_undrifted", seed=seed, fired=fired_un, total=total_un, rate=fa_rate)

            # For contrast: same trigger on an amount-shift stream.
            scenarios = build_default_scenarios(ds.feature_names)
            scenario = next(s for s in scenarios if s.name == "amount_multiplicative_abrupt")
            trigger_dr = PSITrigger(
                baseline_X=X_pre,
                feature_names=ds.feature_names,
                cfg=DriftTriggerConfig(psi_threshold=0.25, min_window_rows=200),
            )
            drifted_X, _drifted_y, _ = make_baseline_stream(X_stream, y_stream, scenario, seed=seed)
            fired_dr, total_dr = _false_alarm_rate(
                trigger_dr, drifted_X, window_size=args.window_size, n_windows=args.n_windows
            )
            dr_rate = fired_dr / max(total_dr, 1)
            drifted_rates.append(dr_rate)
            log.info("true_alert_drifted", seed=seed, fired=fired_dr, total=total_dr, rate=dr_rate)

        # ---------------- Noisy-telemetry AUROC ----------------
        # Train a fresh GNN scorer, then re-score with progressively more
        # edges dropped and record AUROC(root vs rest).
        scorer = GNNResponsibilityScorer.build(
            adapter=adapter,
            baseline_X=X_pre[:20_000],
            feature_names=ds.feature_names,
            baseline_f1=baseline_f1,
            k_overrides={"layer1": cfg.cdag.layer_1_clusters, "layer2": cfg.cdag.layer_2_clusters},
            window_size=512,
            gnn_cfg=GNNConfig(),
        )
        samples = generate_sandbox_dataset(
            adapter=adapter,
            tap=scorer.tap,
            node_set=scorer.node_set,
            baseline_means=scorer.baseline_means,
            baseline_stds=scorer.baseline_stds,
            baseline_stream_X=X_stream,
            baseline_stream_y=y_stream,
            feature_names=ds.feature_names,
            n_samples=args.gnn_samples,
            window_size=512,
            seed=0,
            device=dev,
        )
        train_gnn(
            scorer.model,
            samples,
            cfg=GNNTrainConfig(epochs=args.gnn_epochs),
            device=dev,
        )

        # AUROC per drop_frac, averaged over scenarios × seeds.
        eval_scenarios = build_default_scenarios(ds.feature_names)
        drop_auroc: dict[float, list[float]] = {p: [] for p in args.drop_fracs}
        for seed in range(min(args.seeds, 3)):
            set_global_seed(seed)
            for sc in eval_scenarios:
                stream_X_d, stream_y_d, injector = make_baseline_stream(
                    X_stream, y_stream, sc, seed=seed
                )
                gt = injector.ground_truth().root_cause_feature_idx
                window_X = stream_X_d[: args.window_size * 3]
                window_y = stream_y_d[: args.window_size * 3]
                # Build CDAG once for this scenario/seed.
                probs = adapter.predict_proba(window_X)
                preds = (probs >= adapter.decision_threshold).astype(np.int64)
                current_f1 = float(f1_score(window_y, preds, zero_division=0))
                w_size = min(scorer.window_size, max(1, window_X.shape[0] // 4))
                signals = compute_windowed_signals(
                    window_X,
                    window_y,
                    scorer.tap,
                    scorer.node_set,
                    scorer.baseline_means,
                    scorer.baseline_stds,
                    window_f1=current_f1,
                    window_size=w_size,
                )
                cdag = build_cdag_from_windows(
                    signals, pc_alpha=0.05, notears_lambda=0.05, notears_max_iter=25
                )
                node_features = signals[-1]
                for drop_p in args.drop_fracs:
                    W_perturbed = _drop_edges(
                        cdag.weights, drop_p, np.random.default_rng(seed * 100 + int(drop_p * 100))
                    )
                    data = cdag_to_pyg_data(W_perturbed, node_features, device=dev)
                    scorer.model.eval()
                    with torch.no_grad():
                        logits, _ = scorer.model(data)
                    n_feat = len(scorer.node_set.feature_indices)
                    feat_logits = logits[:n_feat].detach().cpu().numpy()
                    labels = np.zeros(n_feat)
                    labels[gt] = 1
                    try:
                        auroc = float(roc_auc_score(labels, feat_logits))
                    except ValueError:
                        auroc = float("nan")
                    if not np.isnan(auroc):
                        drop_auroc[drop_p].append(auroc)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ---------------- Summarize + print ----------------
        fa = _mean_std(undrifted_rates)
        tr = _mean_std(drifted_rates)
        summary = {
            "false_alarm_rate_undrifted": {"mean": fa[0], "std": fa[1], "values": undrifted_rates},
            "true_alert_rate_drifted": {"mean": tr[0], "std": tr[1], "values": drifted_rates},
            "noisy_telemetry_auroc": {
                str(p): _mean_std(drop_auroc[p]) for p in args.drop_fracs
            },
            "config": vars(args),
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        mlflow.log_artifact(args.out)
        mlflow.log_metric("false_alarm_rate_undrifted_mean", fa[0])
        mlflow.log_metric("true_alert_rate_drifted_mean", tr[0])

        print("\n=== Gate F (robustness) ===")
        print(f"PSI false-alarm rate on UNDRIFTED stream: {fa[0]:.3f} +/- {fa[1]:.3f}")
        print(f"PSI true-alert  rate on DRIFTED   stream: {tr[0]:.3f} +/- {tr[1]:.3f}")
        print("\nGNN attribution AUROC vs edge-drop fraction:")
        for p in args.drop_fracs:
            m, s = summary["noisy_telemetry_auroc"][str(p)]
            print(f"  drop_frac={p:.2f}  AUROC = {m:.4f} +/- {s:.4f}   n={len(drop_auroc[p])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
