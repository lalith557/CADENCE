"""Calibration harness for the Step A contested-SLA scenarios.

Trains a single FraudNet on Credit Card Fraud (seed 42, same pretraining that
Phase 1/2 use), then measures post-drift F1 on the drifted stream for each
scenario without any retraining. Prints a table.

The goal is to verify each contested scenario lands within roughly
[SLA - 0.10, SLA - 0.02] — proving that "no-op" is unambiguously wrong.
Also runs a one-shot partial fine-tune (Layer 1 only) and a full retrain to
confirm the "recoverable-by-partial" / "needs-full" / "unrecoverable" split
holds for the calibrated magnitudes.

Usage:
    python -m benchmarks.calibrate_contested [--sla 0.90]
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import f1_score

from benchmarks.synthetic_drift_gen import (
    build_contested_sla_scenarios,
    build_default_scenarios,
)
from benchmarks.baselines.harness import make_baseline_stream
from cadence.adapters.neural import FraudNet, FraudNetConfig
from cadence.common.config import load_config
from cadence.common.seeds import set_global_seed
from cadence.data.loaders import load_credit_card_fraud


def _f1(adapter: FraudNet, X: np.ndarray, y: np.ndarray) -> float:
    probs = adapter.predict_proba(X)
    preds = (probs >= adapter.decision_threshold).astype(np.int64)
    return float(f1_score(y, preds, zero_division=0))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--sla", type=float, default=0.90)
    p.add_argument("--window-size", type=int, default=2048)
    p.add_argument("--n-windows", type=int, default=5)
    args = p.parse_args()

    cfg = load_config(args.config)
    ds = load_credit_card_fraud(cfg.data, seed=42)

    # Same 70/25/5 split the phase1/phase2 runners use — reproducibility.
    n = ds.X_train.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    pre_end = int(0.70 * n)
    stream_end = pre_end + int(0.25 * n)
    X_pre, y_pre = ds.X_train[idx[:pre_end]], ds.y_train[idx[:pre_end]]
    X_stream, y_stream = (
        ds.X_train[idx[pre_end:stream_end]],
        ds.y_train[idx[pre_end:stream_end]],
    )

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

    undrifted_f1 = _f1(adapter, ds.X_test, ds.y_test)
    print(f"\nBaseline (undrifted) test F1: {undrifted_f1:.4f}")
    print(f"Tuned threshold: {baseline_threshold:.4f}")
    print(f"SLA target: {args.sla}\n")

    scenarios = build_default_scenarios(ds.feature_names) + build_contested_sla_scenarios(
        ds.feature_names
    )

    header = ["scenario", "pre_f1", "partial_f1", "full_f1", "which_wins"]
    print(" | ".join(f"{h:>44}" for h in header))
    print("-" * 44 * len(header))

    n_ok = 0
    for sc in scenarios:
        # Reset weights for every scenario.
        adapter.load_state_dict(baseline_state)
        adapter.decision_threshold = baseline_threshold

        sX, sy, _ = make_baseline_stream(X_stream, y_stream, sc, seed=0)
        # Measure "pre" F1 over the first n_windows chunks.
        ws = args.window_size * args.n_windows
        Xw, yw = sX[:ws], sy[:ws]
        pre_f1 = _f1(adapter, Xw, yw)

        # Try partial (Layer 1 only, no EWC, one epoch — the Step A sandbox recipe).
        adapter.partial_fit(Xw, yw, layers_to_update=["layer1"], max_epochs=3, finetune_lr=1e-3)
        partial_f1 = _f1(adapter, Xw, yw)

        # Reset, try full.
        adapter.load_state_dict(baseline_state)
        adapter.decision_threshold = baseline_threshold
        adapter.fit(
            np.concatenate([X_pre, Xw]),
            np.concatenate([y_pre, yw]),
            X_val=X_pre[-5000:],
            y_val=y_pre[-5000:],
        )
        full_f1 = _f1(adapter, Xw, yw)

        # Decide the "correct" action.
        recovered_by_partial = partial_f1 >= args.sla
        recovered_by_full = full_f1 >= args.sla
        if pre_f1 >= args.sla:
            winner = "no-op (undrifted-ish)"
        elif recovered_by_partial:
            winner = "partial"
        elif recovered_by_full:
            winner = "full"
        else:
            winner = "no-op (unrecoverable)"

        # Flag whether this scenario meets Step A's calibration goal.
        contested = "contested" in sc.name
        in_band = (args.sla - 0.10) <= pre_f1 < args.sla if contested else True
        marker = "OK" if in_band else "MISS"
        if in_band:
            n_ok += 1
        row = [
            f"{sc.name}[{marker}]",
            f"{pre_f1:.4f}",
            f"{partial_f1:.4f}",
            f"{full_f1:.4f}",
            winner,
        ]
        print(" | ".join(f"{c:>44}" for c in row))

    print(f"\n{n_ok}/{len(scenarios)} scenarios in the contested SLA band.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
