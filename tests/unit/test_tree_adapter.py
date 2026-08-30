"""Guards for LightGBMAdapter — Step F generality across model families."""

from __future__ import annotations

import numpy as np
import pytest

from cadence.adapters.tree import LightGBMAdapter, LightGBMConfig


def _synthetic() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n, d = 400, 8
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = ((X[:, 0] + X[:, 1] > 0) & (rng.random(n) < 0.7)).astype(np.int64)
    return X, y


def _fit_small() -> LightGBMAdapter:
    X, y = _synthetic()
    a = LightGBMAdapter(LightGBMConfig(n_estimators=30, num_leaves=15, seed=0))
    a.fit(X[:320], y[:320], X_val=X[320:], y_val=y[320:])
    return a


def test_fit_populates_booster_and_threshold() -> None:
    a = _fit_small()
    assert a.booster is not None
    assert 0.0 <= a.decision_threshold <= 1.0
    assert a._input_dim == 8
    assert a.n_trainable_params() > 0
    assert a.flops_per_forward() > 0


def test_predict_proba_shape_and_range() -> None:
    a = _fit_small()
    X, _ = _synthetic()
    probs = a.predict_proba(X[:20])
    assert probs.shape == (20,)
    assert probs.dtype == np.float32
    assert (probs >= 0).all() and (probs <= 1).all()


def test_forward_with_traces_returns_empty_dict() -> None:
    a = _fit_small()
    X, _ = _synthetic()
    probs, traces = a.forward_with_traces(X[:5])
    assert probs.shape == (5,)
    assert traces == {}


def test_partial_fit_layer_targeted_raises_notimplemented() -> None:
    a = _fit_small()
    X, y = _synthetic()
    with pytest.raises(NotImplementedError, match="no layer-level partial retrain"):
        a.partial_fit(X, y, layers_to_update=["layer1"])


def test_partial_fit_warm_start_updates_booster() -> None:
    a = _fit_small()
    X, y = _synthetic()
    n_trees_before = a.booster.num_trees()
    result = a.partial_fit(X[:100], y[:100], max_epochs=10)
    assert result.epochs_run == 10
    n_trees_after = a.booster.num_trees()
    assert n_trees_after > n_trees_before


def test_clone_produces_identical_predictions() -> None:
    a = _fit_small()
    X, _ = _synthetic()
    original = a.predict_proba(X[:10])
    b = a.clone()
    cloned = b.predict_proba(X[:10])
    np.testing.assert_allclose(original, cloned, atol=1e-6)
    assert b.decision_threshold == a.decision_threshold


def test_state_dict_roundtrip_preserves_predictions() -> None:
    a = _fit_small()
    X, _ = _synthetic()
    original = a.predict_proba(X[:10])
    state = a.state_dict()
    b = LightGBMAdapter(LightGBMConfig(n_estimators=30, num_leaves=15, seed=0))
    b.load_state_dict(state)
    np.testing.assert_allclose(original, b.predict_proba(X[:10]), atol=1e-6)


def test_partial_fit_before_fit_raises() -> None:
    a = LightGBMAdapter(LightGBMConfig())
    X, y = _synthetic()
    with pytest.raises(RuntimeError, match="before fit"):
        a.partial_fit(X, y)
