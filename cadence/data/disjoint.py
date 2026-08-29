"""Disjoint sub-population splits for the H3 forgetting test (fixes W-21).

The original Phase 1 harness measured forgetting on a random hold-out slice
of the *same* distribution as the pretraining data — which meant partial
retrains would look "less forgetful" only for statistical noise reasons,
not because they actually preserved a distinct sub-population.

This module splits Credit Card Fraud into two genuinely disjoint segments:

  * `main_train_X, main_train_y`     — used to pretrain FraudNet, ~70 % of data
  * `main_stream_X, main_stream_y`   — the drifted production stream (~25 %)
  * `disjoint_segment_X/y`           — a genuinely-different sub-population
                                       (~5 %). This is where forgetting is
                                       measured.

The disjoint criterion is configurable:

  * `high_amount`  — rows where `Amount` z-score > 1.5 (extreme-value
                     transactions that the pretraining model rarely sees;
                     partial retrains that touch only Layer 1 should not
                     degrade behaviour here).
  * `late_time`    — rows where `Time` is in the top 10 % (late-in-day
                     transactions; also under-represented at training time).
  * `class_positive` — the tiny minority class (fraud) held out completely,
                     the ultimate "unrelated task" for a fraud classifier.

Each disjoint split is REMOVED from main_train + main_stream, so the
pretraining model has genuinely never seen it. This makes H3 honest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

DisjointCriterion = Literal["high_amount", "late_time", "class_positive"]


@dataclass(frozen=True)
class DisjointSplit:
    main_train_X: np.ndarray
    main_train_y: np.ndarray
    main_stream_X: np.ndarray
    main_stream_y: np.ndarray
    disjoint_X: np.ndarray
    disjoint_y: np.ndarray
    criterion: str
    n_disjoint: int
    disjoint_positive_rate: float


def make_disjoint_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    criterion: DisjointCriterion = "high_amount",
    train_frac: float = 0.70,
    stream_frac: float = 0.25,
    seed: int = 42,
) -> DisjointSplit:
    """Partition (X, y) into pretrain / stream / disjoint segments.

    Disjoint rows are removed from pretrain + stream so the pretrained model
    has zero exposure to them. The remaining (100 % - disjoint_share) is
    split by `train_frac` / `stream_frac`.
    """
    n = X.shape[0]
    if criterion == "high_amount":
        if "Amount" in feature_names:
            f_idx = feature_names.index("Amount")
        else:
            f_idx = X.shape[1] - 1  # convention: last column
        col = X[:, f_idx]
        thresh = np.mean(col) + 1.5 * np.std(col)
        disjoint_mask = col >= thresh
    elif criterion == "late_time":
        if "Time" in feature_names:
            t_idx = feature_names.index("Time")
        else:
            t_idx = 0
        thresh = np.quantile(X[:, t_idx], 0.90)
        disjoint_mask = X[:, t_idx] >= thresh
    elif criterion == "class_positive":
        disjoint_mask = y == 1
    else:
        raise ValueError(f"unknown criterion={criterion!r}")

    disjoint_X = X[disjoint_mask]
    disjoint_y = y[disjoint_mask]
    remain_X = X[~disjoint_mask]
    remain_y = y[~disjoint_mask]

    rng = np.random.default_rng(seed)
    idx = np.arange(remain_X.shape[0])
    rng.shuffle(idx)
    n_train = int(train_frac * remain_X.shape[0])
    n_stream_end = n_train + int(stream_frac * remain_X.shape[0])
    train_idx = idx[:n_train]
    stream_idx = idx[n_train:n_stream_end]

    return DisjointSplit(
        main_train_X=remain_X[train_idx],
        main_train_y=remain_y[train_idx],
        main_stream_X=remain_X[stream_idx],
        main_stream_y=remain_y[stream_idx],
        disjoint_X=disjoint_X,
        disjoint_y=disjoint_y,
        criterion=criterion,
        n_disjoint=int(disjoint_X.shape[0]),
        disjoint_positive_rate=float(np.mean(disjoint_y)) if disjoint_y.size else 0.0,
    )
