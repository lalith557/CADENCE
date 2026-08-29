"""Guards for the Step A contested-SLA scenarios and MultiFeatureShiftSpec."""

from __future__ import annotations

import numpy as np

from benchmarks.synthetic_drift_gen import (
    DriftInjector,
    MultiFeatureShiftSpec,
    build_contested_sla_scenarios,
    build_default_scenarios,
    build_step_a_scenarios,
)


def _fake_features(n: int = 30) -> list[str]:
    names = ["Time"] + [f"V{i}" for i in range(1, n - 1)] + ["Amount"]
    return names


def test_build_contested_sla_scenarios_has_three_named() -> None:
    names = {sc.name for sc in build_contested_sla_scenarios(_fake_features())}
    assert names == {
        "contested_v14_partial_recoverable",
        "contested_multi_feature_full_needed",
        "contested_v14_tail_flip_unrecoverable",
    }


def test_build_step_a_scenarios_covers_at_least_five() -> None:
    sc = build_step_a_scenarios(_fake_features())
    # Gate A requires >= 5 scenarios.
    assert len(sc) >= 5
    default_names = {s.name for s in build_default_scenarios(_fake_features())}
    contested_names = {s.name for s in build_contested_sla_scenarios(_fake_features())}
    got = {s.name for s in sc}
    assert default_names.issubset(got)
    assert contested_names.issubset(got)


def test_multi_feature_shift_spec_shifts_all_named_indices() -> None:
    feature_names = _fake_features()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((256, len(feature_names))).astype(np.float32)
    y = (rng.random(256) < 0.1).astype(np.int64)

    from benchmarks.synthetic_drift_gen import DriftScenario, DriftSchedule

    sc = DriftScenario(
        name="test_multi",
        spec=MultiFeatureShiftSpec(
            feature_idx=29,
            other_indices=[14],
            multiplicative=2.0,
            additive=0.0,
            other_multiplicative=1.0,
            other_additive=3.0,
        ),
        schedule=DriftSchedule.ABRUPT,
        start_step=0,
        full_effect_step=0,
        feature_names=feature_names,
    )
    injector = DriftInjector(sc)
    X_drift, y_drift = injector.apply(X.copy(), y.copy(), step_offset=0)

    # Amount (idx 29) doubled.
    np.testing.assert_allclose(X_drift[:, 29], 2.0 * X[:, 29], atol=1e-5)
    # V14 (idx 14) shifted up by 3.
    np.testing.assert_allclose(X_drift[:, 14], X[:, 14] + 3.0, atol=1e-5)
    # Other features unchanged.
    np.testing.assert_allclose(X_drift[:, 0], X[:, 0], atol=1e-5)


def test_multi_feature_ground_truth_names_primary_root_cause() -> None:
    feature_names = _fake_features()
    from benchmarks.synthetic_drift_gen import DriftScenario, DriftSchedule

    sc = DriftScenario(
        name="test_multi_gt",
        spec=MultiFeatureShiftSpec(
            feature_idx=29,
            other_indices=[14, 7],
            multiplicative=1.5,
            additive=0.0,
            other_multiplicative=1.0,
            other_additive=1.0,
        ),
        schedule=DriftSchedule.ABRUPT,
        start_step=0,
        full_effect_step=0,
        feature_names=feature_names,
    )
    injector = DriftInjector(sc)
    injector.apply(
        np.zeros((10, len(feature_names)), dtype=np.float32),
        np.zeros(10, dtype=np.int64),
        step_offset=0,
    )
    gt = injector.ground_truth()
    assert gt.root_cause_feature_idx == 29
    assert gt.root_cause_feature_name == "Amount"
    assert gt.mechanism == "covariate_shift"
