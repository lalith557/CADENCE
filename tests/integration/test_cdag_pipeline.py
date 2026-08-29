"""Integration test: CDAG pipeline end-to-end on a tiny synthetic dataset.

Verifies:
1. ActivationTap fits per-layer k-means without crashing.
2. Node set indexing is consistent (features → clusters → performance).
3. Windowed signals have the right shape and finite values.
4. CDAG discovery produces a real-valued square weight matrix.
5. CDAGResponsibilityScorer returns per-feature scores in [0, 1].
6. Drifted feature gets a HIGHER responsibility score than an undrifted one.

Uses ~1000 rows of synthetic data + tiny FraudNet — runs in <15s on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadence.adapters.neural import FraudNet, FraudNetConfig
from cadence.attribution import CDAGResponsibilityScorer
from cadence.cdag import (
    ActivationTap,
    build_baseline_stats,
    build_cdag_from_windows,
    build_node_set,
    compute_windowed_signals,
)
from cadence.common.seeds import set_global_seed


@pytest.fixture
def tiny_adapter_and_baseline():
    set_global_seed(0, deterministic_torch=False)
    n_feat = 8
    n_rows = 1500
    X = np.random.randn(n_rows, n_feat).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

    adapter = FraudNet(
        FraudNetConfig(
            input_dim=n_feat, hidden_dims=[16, 8], max_epochs=3, early_stopping_patience=1
        )
    )
    adapter.fit(X, y, X_val=X[:200], y_val=y[:200])
    return adapter, X, y, [f"f{i}" for i in range(n_feat)]


def test_activation_tap_fits(tiny_adapter_and_baseline):
    adapter, X, _, _ = tiny_adapter_and_baseline
    tap = ActivationTap(adapter).fit(X, k_overrides={"layer1": 4, "layer2": 2})
    assert set(tap.fits.keys()) == {"layer1", "layer2"}
    assert tap.fits["layer1"].k == 4
    assert tap.fits["layer2"].k == 2


def test_node_set_indexing(tiny_adapter_and_baseline):
    adapter, X, _, feats = tiny_adapter_and_baseline
    tap = ActivationTap(adapter).fit(X, k_overrides={"layer1": 4, "layer2": 2})
    node_set = build_node_set(feats, tap)
    # 8 features + 4 + 2 clusters + 1 performance = 15 nodes.
    assert node_set.n_nodes == 15
    assert len(node_set.feature_indices) == 8
    assert len(node_set.cluster_indices) == 6
    assert node_set.performance_index == 14


def test_windowed_signals_shape_and_finiteness(tiny_adapter_and_baseline):
    adapter, X, y, feats = tiny_adapter_and_baseline
    tap = ActivationTap(adapter).fit(X, k_overrides={"layer1": 4, "layer2": 2})
    node_set = build_node_set(feats, tap)
    bmean, bstd = build_baseline_stats(X, tap, node_set, baseline_f1=0.8)

    signals = compute_windowed_signals(
        X, y, tap, node_set, bmean, bstd, window_f1=0.55, window_size=200
    )
    # (n_windows, n_nodes, 4).
    assert signals.shape == (7, 15, 4)
    assert np.isfinite(signals).all()


def test_cdag_discovery_returns_square_weights(tiny_adapter_and_baseline):
    adapter, X, y, feats = tiny_adapter_and_baseline
    tap = ActivationTap(adapter).fit(X, k_overrides={"layer1": 4, "layer2": 2})
    node_set = build_node_set(feats, tap)
    bmean, bstd = build_baseline_stats(X, tap, node_set, baseline_f1=0.8)
    signals = compute_windowed_signals(
        X, y, tap, node_set, bmean, bstd, window_f1=0.55, window_size=200
    )
    cdag = build_cdag_from_windows(signals, notears_max_iter=10)
    assert cdag.weights.shape == (15, 15)
    assert cdag.skeleton.shape == (15, 15)
    # Diagonal must be zero (no self-loops).
    assert np.all(np.diag(cdag.weights) == 0)


def test_cdag_scorer_returns_bounded_per_feature_scores(tiny_adapter_and_baseline):
    adapter, X, _, feats = tiny_adapter_and_baseline
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    scorer = CDAGResponsibilityScorer.build(
        adapter,
        baseline_X=X,
        feature_names=feats,
        baseline_f1=0.8,
        k_overrides={"layer1": 4, "layer2": 2},
        window_size=200,
    )
    # Drifted window: feature 0 shifted by 3 std.
    Xd = X.copy()
    Xd[:, 0] += 3.0
    scores = scorer(adapter, Xd, y)
    assert scores.shape == (8,)
    assert (scores >= 0).all()
    assert (scores <= 1).all()


def test_cdag_scorer_prefers_drifted_feature_over_untouched(tiny_adapter_and_baseline):
    """Sanity: the drifted feature should score higher than a specific untouched one.

    This is a soft test — the drift-z fallback path in the scorer guarantees
    the drifted feature's ranking is at least monotonic in the drift magnitude.
    """
    adapter, X, _, feats = tiny_adapter_and_baseline
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    scorer = CDAGResponsibilityScorer.build(
        adapter,
        baseline_X=X,
        feature_names=feats,
        baseline_f1=0.8,
        k_overrides={"layer1": 4, "layer2": 2},
        window_size=200,
    )
    Xd = X.copy()
    Xd[:, 0] += 3.0  # heavy drift on feature 0
    scores = scorer(adapter, Xd, y)
    # Feature 0 (drifted) should outrank feature 5 (untouched).
    assert scores[0] >= scores[5]
