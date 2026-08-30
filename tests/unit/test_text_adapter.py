"""Guards for TFIDFLogRegAdapter — Step F text-drift generality."""

from __future__ import annotations

import numpy as np
import pytest

from cadence.adapters.text import TextConfig, TFIDFLogRegAdapter


def _synthetic_text() -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(0)
    pos_terms = ["great food friendly service", "amazing atmosphere wonderful experience"]
    neg_terms = ["terrible awful service", "bad experience never again"]
    texts, y = [], []
    for _ in range(60):
        if rng.random() < 0.5:
            texts.append(rng.choice(pos_terms) + f" run {rng.integers(0, 100)}")
            y.append(1)
        else:
            texts.append(rng.choice(neg_terms) + f" run {rng.integers(0, 100)}")
            y.append(0)
    return texts, np.array(y, dtype=np.int64)


def _fit_small() -> TFIDFLogRegAdapter:
    a = TFIDFLogRegAdapter(TextConfig(max_features=100, min_df=1))
    X, y = _synthetic_text()
    a.fit(X[:48], y[:48], X_val=X[48:], y_val=y[48:])
    return a


def test_fit_freezes_vocabulary() -> None:
    a = _fit_small()
    vocab_before = set(a.vectorizer.vocabulary_.keys())
    # A partial_fit adds new documents (with new tokens); vocab stays frozen.
    # Provide both classes so LogisticRegression can actually fit.
    new_texts = ["completely unseen unique token abcdef"] * 10 + [
        "another totally different vocabulary xyz"
    ] * 10
    new_y = np.array([1] * 10 + [0] * 10)
    a.partial_fit(new_texts, new_y)
    vocab_after = set(a.vectorizer.vocabulary_.keys())
    assert vocab_before == vocab_after, "vocabulary must be frozen after fit"


def test_forward_with_traces_returns_empty_dict() -> None:
    a = _fit_small()
    probs, traces = a.forward_with_traces(["ok food good service"])
    assert probs.shape == (1,)
    assert traces == {}


def test_partial_fit_layer_targeted_raises_notimplemented() -> None:
    a = _fit_small()
    X, y = _synthetic_text()
    with pytest.raises(NotImplementedError, match="no layer-level"):
        a.partial_fit(X, y, layers_to_update=["layer1"])


def test_state_dict_roundtrip_preserves_predictions() -> None:
    a = _fit_small()
    X, _ = _synthetic_text()
    original = a.predict_proba(X[:8])
    state = a.state_dict()
    b = TFIDFLogRegAdapter(TextConfig(max_features=100, min_df=1))
    b.load_state_dict(state)
    np.testing.assert_allclose(original, b.predict_proba(X[:8]), atol=1e-6)


def test_partial_fit_before_fit_raises() -> None:
    a = TFIDFLogRegAdapter(TextConfig())
    with pytest.raises(RuntimeError, match="before fit"):
        a.partial_fit(["some text"], np.array([1]))


def test_clone_produces_identical_predictions() -> None:
    a = _fit_small()
    X, _ = _synthetic_text()
    original = a.predict_proba(X[:8])
    b = a.clone()
    np.testing.assert_allclose(original, b.predict_proba(X[:8]), atol=1e-6)
