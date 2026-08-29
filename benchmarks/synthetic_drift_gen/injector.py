"""Synthetic drift injector with fully-logged ground-truth causes.

This is the **answer key** for H1 attribution experiments. Every drift we inject
records:
  - the specific input-feature index it targeted (`root_cause_feature`)
  - the mechanism (feature shift vs concept shift)
  - the magnitude
  - the step at which it started + whether it's abrupt/gradual

Later, when CDAG says "responsibility node = X", we compare X against
`root_cause_feature` to get attribution precision/recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np


class DriftSchedule(str, Enum):
    ABRUPT = "abrupt"
    GRADUAL = "gradual"


@dataclass
class FeatureShiftSpec:
    """Shift ONE feature's distribution: X[:, f] -> a * X[:, f] + b.

    Ground truth: this feature is the root cause; the concept mapping y=f(x) is
    UNCHANGED, so the model's error comes purely from covariate shift.
    """

    feature_idx: int
    multiplicative: float = 1.0
    additive: float = 0.0

    def kind(self) -> Literal["covariate_shift"]:
        return "covariate_shift"


@dataclass
class ConceptShiftSpec:
    """Flip labels in a region: whenever X[:, f] crosses threshold, invert y.

    Ground truth: `feature_idx` is the root cause; the concept mapping itself
    changed (a form of concept drift).
    """

    feature_idx: int
    threshold: float
    flip_probability: float = 1.0

    def kind(self) -> Literal["concept_shift"]:
        return "concept_shift"


@dataclass
class MultiFeatureShiftSpec:
    """Shift several features simultaneously.

    Used for the "needs-full-retrain" contested scenario in Step A: a single-
    layer partial retrain can nudge one feature's activation cluster back but
    can't fix a drift that touches multiple orthogonal feature dimensions at
    once — that's exactly what a full retrain is for.

    Ground truth: `feature_idx` names the *primary* root cause (the strongest
    single contributor). Attribution scorers should still hit this one, but the
    presence of others is the reason partial won't be enough.
    """

    feature_idx: int
    other_indices: list[int] = field(default_factory=list)
    multiplicative: float = 1.0
    additive: float = 0.0
    other_multiplicative: float = 1.0
    other_additive: float = 0.0

    def kind(self) -> Literal["covariate_shift"]:
        return "covariate_shift"


DriftSpec = FeatureShiftSpec | ConceptShiftSpec | MultiFeatureShiftSpec


@dataclass
class DriftScenario:
    """A single, named drift scenario for the benchmark."""

    name: str
    spec: DriftSpec
    schedule: DriftSchedule
    start_step: int
    full_effect_step: (
        int  # for gradual: linear ramp from start_step→here; for abrupt: same as start
    )
    feature_names: list[str] = field(default_factory=list)


@dataclass
class InjectedGroundTruth:
    """What actually got applied to the data — logged for scoring."""

    scenario_name: str
    root_cause_feature_idx: int
    root_cause_feature_name: str
    mechanism: Literal["covariate_shift", "concept_shift"]
    schedule: str
    magnitude: float
    n_rows_affected: int


class DriftInjector:
    """Apply a `DriftScenario` to a stream of (X, y) batches, logging ground truth."""

    def __init__(self, scenario: DriftScenario, rng: np.random.Generator | None = None):
        self.scenario = scenario
        self.rng = rng or np.random.default_rng(0)
        self._n_rows_processed = 0
        self._n_rows_affected = 0
        self._magnitude_total = 0.0

    def _intensity_at_step(self, step: int) -> float:
        """Return effect intensity in [0, 1] at the given global step."""
        s = self.scenario
        if step < s.start_step:
            return 0.0
        if s.schedule == DriftSchedule.ABRUPT:
            return 1.0
        # gradual: linear ramp
        if step >= s.full_effect_step:
            return 1.0
        span = max(1, s.full_effect_step - s.start_step)
        return (step - s.start_step) / span

    def apply(
        self, X: np.ndarray, y: np.ndarray, *, step_offset: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply drift to a batch. `step_offset` is the global step at which THIS
        batch's first row lands, so we can implement gradual ramps across batches."""
        X = X.copy()
        y = y.copy()
        n = X.shape[0]
        # Compute per-row step for gradual ramps.
        steps = np.arange(step_offset, step_offset + n)
        intensity = np.array([self._intensity_at_step(int(s)) for s in steps], dtype=np.float32)
        active = intensity > 0

        spec = self.scenario.spec
        if isinstance(spec, FeatureShiftSpec):
            f = spec.feature_idx
            X[active, f] = ((spec.multiplicative - 1.0) * intensity[active] + 1.0) * X[
                active, f
            ] + spec.additive * intensity[active]
            self._magnitude_total += float(
                np.mean(
                    np.abs(
                        (spec.multiplicative - 1.0) * intensity[active]
                        + spec.additive * intensity[active]
                    )
                )
                if active.any()
                else 0.0
            )
        elif isinstance(spec, ConceptShiftSpec):
            f = spec.feature_idx
            crosses = X[:, f] >= spec.threshold
            # With probability p * intensity, flip the label.
            flip_prob = spec.flip_probability * intensity
            flips = (self.rng.random(n) < flip_prob) & crosses & active
            y[flips] = 1 - y[flips]
            self._magnitude_total += float(np.mean(flip_prob[active]) if active.any() else 0.0)
        elif isinstance(spec, MultiFeatureShiftSpec):
            f = spec.feature_idx
            X[active, f] = ((spec.multiplicative - 1.0) * intensity[active] + 1.0) * X[
                active, f
            ] + spec.additive * intensity[active]
            for other in spec.other_indices:
                X[active, other] = ((spec.other_multiplicative - 1.0) * intensity[active] + 1.0) * X[
                    active, other
                ] + spec.other_additive * intensity[active]
            self._magnitude_total += float(
                np.mean(np.abs(intensity[active])) if active.any() else 0.0
            ) * (1 + len(spec.other_indices))
        else:  # pragma: no cover
            raise TypeError(f"unknown drift spec: {type(spec)!r}")

        self._n_rows_processed += n
        self._n_rows_affected += int(active.sum())
        return X, y

    def ground_truth(self) -> InjectedGroundTruth:
        s = self.scenario
        spec = s.spec
        if isinstance(spec, FeatureShiftSpec):
            magnitude = abs(spec.multiplicative - 1.0) + abs(spec.additive)
            mechanism: Literal["covariate_shift", "concept_shift"] = "covariate_shift"
        elif isinstance(spec, MultiFeatureShiftSpec):
            magnitude = abs(spec.multiplicative - 1.0) + abs(spec.additive) + len(
                spec.other_indices
            ) * (abs(spec.other_multiplicative - 1.0) + abs(spec.other_additive))
            mechanism = "covariate_shift"
        else:
            magnitude = spec.flip_probability
            mechanism = "concept_shift"
        feat_name = (
            s.feature_names[spec.feature_idx]
            if 0 <= spec.feature_idx < len(s.feature_names)
            else f"feature_{spec.feature_idx}"
        )
        return InjectedGroundTruth(
            scenario_name=s.name,
            root_cause_feature_idx=spec.feature_idx,
            root_cause_feature_name=feat_name,
            mechanism=mechanism,
            schedule=s.schedule.value,
            magnitude=magnitude,
            n_rows_affected=self._n_rows_affected,
        )


# ---------- canonical scenario library ----------


def build_default_scenarios(feature_names: list[str]) -> list[DriftScenario]:
    """The set the paper reports on for the Credit Card Fraud dataset.

    Chosen so each targets a different mechanism × schedule × feature location
    (early feature vs late feature), giving us a spread that avoids
    overfitting attribution to a single scenario shape.
    """
    n = len(feature_names)
    # Indices in the Fraud dataset: Time=0, V1..V28=1..28, Amount=29.
    # Fall back to safe indices when a synthetic tiny feature space is used
    # (e.g., unit tests with 8 features).
    amount_idx = feature_names.index("Amount") if "Amount" in feature_names else min(29, n - 1)
    v14_idx = feature_names.index("V14") if "V14" in feature_names else min(14, n - 1)
    time_idx = feature_names.index("Time") if "Time" in feature_names else 0

    return [
        DriftScenario(
            name="amount_multiplicative_abrupt",
            spec=FeatureShiftSpec(feature_idx=amount_idx, multiplicative=1.5, additive=0.0),
            schedule=DriftSchedule.ABRUPT,
            start_step=0,
            full_effect_step=0,
            feature_names=feature_names,
        ),
        DriftScenario(
            name="amount_multiplicative_gradual",
            spec=FeatureShiftSpec(feature_idx=amount_idx, multiplicative=2.0, additive=0.0),
            schedule=DriftSchedule.GRADUAL,
            start_step=0,
            full_effect_step=10_000,
            feature_names=feature_names,
        ),
        DriftScenario(
            name="v14_additive_abrupt",
            spec=FeatureShiftSpec(feature_idx=v14_idx, multiplicative=1.0, additive=1.5),
            schedule=DriftSchedule.ABRUPT,
            start_step=0,
            full_effect_step=0,
            feature_names=feature_names,
        ),
        DriftScenario(
            name="v14_concept_shift",
            spec=ConceptShiftSpec(feature_idx=v14_idx, threshold=0.5, flip_probability=0.8),
            schedule=DriftSchedule.ABRUPT,
            start_step=0,
            full_effect_step=0,
            feature_names=feature_names,
        ),
        DriftScenario(
            name="time_gradual",
            spec=FeatureShiftSpec(feature_idx=time_idx, multiplicative=1.2, additive=0.5),
            schedule=DriftSchedule.GRADUAL,
            start_step=2000,
            full_effect_step=8000,
            feature_names=feature_names,
        ),
    ]


def build_contested_sla_scenarios(feature_names: list[str]) -> list[DriftScenario]:
    """The Step A scenarios: post-drift F1 sits *just* below SLA (0.90) so
    "do nothing" is provably wrong on every seed.

    Three scenarios, chosen so each of {partial, full, no-op} is the correct
    action somewhere:

      (a) `contested_amount_mild_partial_recoverable`
          A modest multiplicative shift on `Amount` (a shallow feature that
          Layer-1's cluster handles). Partial retrain of Layer 1 recovers
          fully → **partial is the correct action**.

      (b) `contested_multi_feature_full_needed`
          Two-feature simultaneous shift on `Amount` + `V14`. A Layer-1-only
          partial nudges the shallow signal back but leaves the deeper
          representation broken; only a full retrain restores F1 → **full is
          the correct action**.

      (c) `contested_v14_hard_flip_unrecoverable`
          Concept-flip on V14 above threshold, high flip probability. The old
          concept is genuinely gone; neither partial nor full recovers F1
          without new labels. Burning compute on retrains is wrong → **no-op
          is the correct action (escalate a human instead)**.

    Every scenario is deliberately calibrated so post-drift F1 lands in
    roughly [SLA - 0.10, SLA - 0.02]; the naïve "no-op-forever" policy that
    won Phase 2's bang-bang search cannot win these scenarios.
    """
    n = len(feature_names)
    amount_idx = feature_names.index("Amount") if "Amount" in feature_names else min(29, n - 1)
    v14_idx = feature_names.index("V14") if "V14" in feature_names else min(14, n - 1)

    return [
        DriftScenario(
            name="contested_v14_partial_recoverable",
            # V14 is a strong PCA feature FraudNet actually uses. A moderate
            # additive shift on V14 knocks F1 below the SLA=0.90 × baseline
            # floor but a Layer-1 partial retrain of the L1 cluster tied to
            # V14 can recover most of it. Correct action: partial.
            spec=FeatureShiftSpec(feature_idx=v14_idx, multiplicative=1.0, additive=1.0),
            schedule=DriftSchedule.ABRUPT,
            start_step=0,
            full_effect_step=0,
            feature_names=feature_names,
        ),
        DriftScenario(
            name="contested_multi_feature_full_needed",
            # Two-feature shift: Layer-1-only partial can't recover, full does.
            spec=MultiFeatureShiftSpec(
                feature_idx=amount_idx,
                other_indices=[v14_idx],
                multiplicative=1.20,
                additive=0.0,
                other_multiplicative=0.5,
                other_additive=-0.75,
            ),
            schedule=DriftSchedule.ABRUPT,
            start_step=0,
            full_effect_step=0,
            feature_names=feature_names,
        ),
        DriftScenario(
            name="contested_v14_tail_flip_unrecoverable",
            # Flip labels only where V14 is in the moderate tail (>= 1.0).
            # A meaningful fraction of positives get relabeled, but the
            # feature-space distribution itself is unchanged, so no amount of
            # partial/full retraining on old-labels-plus-new-window data
            # recovers the SLA. Correct action: no-op + escalate a human.
            spec=ConceptShiftSpec(feature_idx=v14_idx, threshold=1.0, flip_probability=0.5),
            schedule=DriftSchedule.ABRUPT,
            start_step=0,
            full_effect_step=0,
            feature_names=feature_names,
        ),
    ]


def build_step_a_scenarios(feature_names: list[str]) -> list[DriftScenario]:
    """Union set used by Step A / Gate A — includes both the existing scenarios
    (so we don't lose any previous R-4 comparability) and the contested three
    where "do nothing" is unambiguously wrong. ≥5 scenarios in total, matching
    the Gate A protocol.
    """
    base = build_default_scenarios(feature_names)
    contested = build_contested_sla_scenarios(feature_names)
    return base + contested
