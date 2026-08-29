"""Guards for the §4 fallbacks + escalation ladder.

One test per fallback (§4.1-§4.8), each forcing the failure it protects
against and asserting the fallback fires.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadence.data.disjoint import make_disjoint_split
from cadence.executor import (
    DiffuseFallbackConfig,
    EscalationConfig,
    EscalationDecision,
    EscalationLadder,
    ExecutorConfig,
    FalseAlarmConfig,
    ResourceFallbackError,
    RetrainExecutor,
    cold_start_scores,
    diffuse_responsibility_fallback,
    herfindahl_concentration,
    is_false_alarm,
    resource_fallback,
)


# ---------- §4.4 diffuse-responsibility fallback ----------


def test_diffuse_fallback_escalates_partial_to_full_on_flat_scores() -> None:
    flat_scores = np.full(30, 0.5, dtype=np.float32)  # every feature "responsible"
    out = diffuse_responsibility_fallback("partial", flat_scores)
    assert out == "full"


def test_diffuse_fallback_leaves_partial_alone_when_concentrated() -> None:
    peaked = np.zeros(30, dtype=np.float32)
    peaked[7] = 1.0
    out = diffuse_responsibility_fallback("partial", peaked)
    assert out == "partial"


def test_diffuse_fallback_does_not_touch_noop_or_full() -> None:
    flat = np.full(30, 0.1, dtype=np.float32)
    assert diffuse_responsibility_fallback("no-op", flat) == "no-op"
    assert diffuse_responsibility_fallback("full", flat) == "full"


def test_herfindahl_matches_expected_extremes() -> None:
    # Fully concentrated -> 1.0
    peaked = np.zeros(10)
    peaked[3] = 5.0
    assert herfindahl_concentration(peaked) == pytest.approx(1.0)
    # Uniform -> 0.0 (rescaled)
    uniform = np.ones(10)
    assert herfindahl_concentration(uniform) == pytest.approx(0.0, abs=1e-6)


# ---------- §4.6 false-alarm handling ----------


def test_false_alarm_suppressed_when_delayed_f1_above_sla() -> None:
    assert is_false_alarm(True, 0.90, cfg=FalseAlarmConfig(sla_target=0.65))


def test_false_alarm_not_suppressed_when_delayed_f1_below_sla() -> None:
    assert not is_false_alarm(True, 0.40, cfg=FalseAlarmConfig(sla_target=0.65))


def test_false_alarm_ignored_when_psi_did_not_fire() -> None:
    assert not is_false_alarm(False, 0.90)


def test_false_alarm_ignored_when_labels_missing() -> None:
    # Delayed labels not yet arrived -> assume alert is legitimate.
    assert not is_false_alarm(True, None)


# ---------- §4.7 resource fallback ----------


def test_resource_fallback_re_raises_as_resource_error_on_oom() -> None:
    import torch

    def _fail():
        raise torch.cuda.OutOfMemoryError("simulated OOM")

    with pytest.raises(ResourceFallbackError, match="OOM"):
        resource_fallback(_fail)


def test_resource_fallback_re_raises_on_timeout() -> None:
    def _fail():
        raise TimeoutError("simulated timeout")

    with pytest.raises(ResourceFallbackError, match="timeout"):
        resource_fallback(_fail)


def test_resource_fallback_returns_value_on_success() -> None:
    assert resource_fallback(lambda: 42) == 42


def test_resource_fallback_lets_unrelated_exceptions_propagate() -> None:
    with pytest.raises(ValueError, match="unrelated"):
        resource_fallback(lambda: (_ for _ in ()).throw(ValueError("unrelated")))


# ---------- §4.8 cold-start / missing telemetry ----------


def test_cold_start_uses_feature_only_when_activations_missing() -> None:
    feature_scores = np.arange(5, dtype=np.float32)
    out = cold_start_scores(activation_traces=None, feature_scores=feature_scores, n_features=5)
    np.testing.assert_array_equal(out, feature_scores)


def test_cold_start_falls_back_to_uniform_when_all_missing() -> None:
    out = cold_start_scores(activation_traces=None, feature_scores=None, n_features=10)
    assert out.shape == (10,)
    np.testing.assert_allclose(out, 0.1)


def test_cold_start_passthrough_when_all_present() -> None:
    fake_traces = {"layer1": np.zeros((3, 4))}
    fs = np.arange(5, dtype=np.float32)
    out = cold_start_scores(activation_traces=fake_traces, feature_scores=fs, n_features=5)
    np.testing.assert_array_equal(out, fs)


# ---------- Disjoint sub-population (W-21) ----------


def test_disjoint_split_high_amount_removes_tail_from_train() -> None:
    rng = np.random.default_rng(0)
    n = 400
    X = rng.standard_normal((n, 30)).astype(np.float32)
    # Force clear high-Amount tail.
    X[:20, -1] = 100.0
    y = (rng.random(n) < 0.1).astype(np.int64)
    feature_names = [f"V{i}" for i in range(1, 30)] + ["Amount"]
    split = make_disjoint_split(X, y, feature_names, criterion="high_amount", seed=0)
    # Disjoint rows are the extreme tail.
    assert split.disjoint_X.shape[0] >= 15
    # Extreme values should not leak into main_train_X.
    assert split.main_train_X[:, -1].max() < X[:20, -1].min()


def test_disjoint_split_class_positive_isolates_fraud() -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 5)).astype(np.float32)
    y = np.zeros(200, dtype=np.int64)
    y[:12] = 1
    feature_names = [f"V{i}" for i in range(5)]
    split = make_disjoint_split(X, y, feature_names, criterion="class_positive", seed=0)
    assert split.disjoint_y.sum() == 12
    assert split.main_train_y.sum() == 0
    assert split.main_stream_y.sum() == 0


# ---------- Escalation ladder (§4.3 + §4.5) ----------


class _StubExecutor:
    """Minimal executor stub: returns rolled-back events until `promote_on_action`."""

    def __init__(self, promote_on_action: str | None = None) -> None:
        from cadence.executor.executor import ExecutorEvent, ExecutorReport

        self._promote = promote_on_action
        self._ExecutorEvent = ExecutorEvent
        self._ExecutorReport = ExecutorReport
        self.calls: list[str] = []

    def execute(self, *, action, window_X, window_y, target_layer=None, predicted_post_f1=None):
        self.calls.append(action)
        rolled = action != self._promote
        event = self._ExecutorEvent(
            stage="rolled_back" if rolled else "promoted",
            action=action,
            target_layer=target_layer,
            pre_f1=0.5,
            post_f1=0.4 if rolled else 0.9,
            shadow_f1=0.3 if rolled else 0.9,
            predicted_post_f1=predicted_post_f1,
            passed_shadow=not rolled,
            was_rolled_back=rolled,
            gpu_seconds=0.0,
            kg_co2=0.0,
            wall_seconds=0.0,
        )
        return self._ExecutorReport(action_taken=action, event=event, surrogate_feedback=None)


def test_escalation_ladder_stops_at_first_success() -> None:
    ladder = EscalationLadder(EscalationConfig(ladder=("partial", "full"), unrecoverable_streak=3))
    exec_ = _StubExecutor(promote_on_action="partial")
    result = ladder.run_ladder(
        exec_, window_X=np.zeros((4, 2)), window_y=np.zeros(4), target_layer_for_partial="layer1"
    )
    assert result.decision == EscalationDecision.PROMOTED
    assert exec_.calls == ["partial"]


def test_escalation_ladder_tries_full_after_partial_fails() -> None:
    ladder = EscalationLadder(EscalationConfig(ladder=("partial", "full"), unrecoverable_streak=3))
    exec_ = _StubExecutor(promote_on_action="full")
    result = ladder.run_ladder(
        exec_, window_X=np.zeros((4, 2)), window_y=np.zeros(4), target_layer_for_partial="layer1"
    )
    assert result.decision == EscalationDecision.PROMOTED
    assert exec_.calls == ["partial", "full"]


def test_escalation_ladder_flags_unrecoverable_after_streak() -> None:
    ladder = EscalationLadder(EscalationConfig(ladder=("partial", "full"), unrecoverable_streak=2))
    exec_ = _StubExecutor(promote_on_action=None)  # nothing promotes

    # First failure -> ESCALATED but not yet unrecoverable.
    r1 = ladder.run_ladder(
        exec_, window_X=np.zeros((4, 2)), window_y=np.zeros(4), target_layer_for_partial="layer1"
    )
    assert r1.decision == EscalationDecision.ESCALATED_HUMAN
    assert not ladder.is_unrecoverable

    # Second failure -> hits threshold, becomes unrecoverable.
    r2 = ladder.run_ladder(
        exec_, window_X=np.zeros((4, 2)), window_y=np.zeros(4), target_layer_for_partial="layer1"
    )
    assert r2.decision == EscalationDecision.ESCALATED_HUMAN
    assert ladder.is_unrecoverable

    # Third call short-circuits (no more compute burned).
    prev_calls = len(exec_.calls)
    r3 = ladder.run_ladder(
        exec_, window_X=np.zeros((4, 2)), window_y=np.zeros(4), target_layer_for_partial="layer1"
    )
    assert r3.decision == EscalationDecision.ESCALATED_HUMAN
    assert len(exec_.calls) == prev_calls  # no new executor calls


# ---------- Executor rollback (§4.1 + §4.2) ----------


def test_executor_no_op_returns_immediately() -> None:
    from cadence.adapters.neural import FraudNet, FraudNetConfig

    adapter = FraudNet(FraudNetConfig(input_dim=4, hidden_dims=[8]))
    # Fit briefly so predict_proba works.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 4)).astype(np.float32)
    y = (rng.random(64) < 0.3).astype(np.int64)
    adapter.fit(X, y, X_val=X[-16:], y_val=y[-16:])

    exec_ = RetrainExecutor(adapter, historical_X=X, historical_y=y, cfg=ExecutorConfig())
    report = exec_.execute(action="no-op", window_X=X[:16], window_y=y[:16])
    assert report.action_taken == "no-op"
    assert not report.event.was_rolled_back
    assert report.event.gpu_seconds == 0.0
