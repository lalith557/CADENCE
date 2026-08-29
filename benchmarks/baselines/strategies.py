"""Baseline strategies for Phase 1: PSI+full-retrain, fixed-schedule, EWC-only.

Each `Strategy` implements `decide(...)` and returns one of:
  - "no-op"                 — do nothing this window
  - "full"                  — full retrain
  - "partial_all_layers"    — EWC-regularized fine-tune of ALL layers (no
                              causal-targeted freezing) — the honest "EWC-only"
                              baseline that lets us separate the value of EWC
                              itself from the value of CDAG-driven targeting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cadence.adapters.base import ModelAdapter
from cadence.collector.drift_trigger import PSITrigger


@dataclass
class PSIFullRetrainStrategy:
    """Baseline A — the mainstream MLOps default: PSI alert → full retrain."""

    name: str = "psi_full_retrain"
    cooldown_windows: int = 1
    _last_retrain: int = -999

    def decide(
        self,
        *,
        window_idx: int,
        adapter: ModelAdapter,
        window_X: np.ndarray,
        window_y: np.ndarray,
        trigger: PSITrigger,
        rolling_f1: float,
    ) -> str:
        alert = trigger.evaluate(window_X)
        if alert.fired and (window_idx - self._last_retrain) >= self.cooldown_windows:
            self._last_retrain = window_idx
            return "full"
        return "no-op"


@dataclass
class FixedScheduleStrategy:
    """Baseline B — retrain every N windows regardless of drift."""

    name: str = "fixed_schedule"
    period_windows: int = 5

    def decide(
        self,
        *,
        window_idx: int,
        adapter: ModelAdapter,
        window_X: np.ndarray,
        window_y: np.ndarray,
        trigger: PSITrigger,
        rolling_f1: float,
    ) -> str:
        if window_idx > 0 and window_idx % self.period_windows == 0:
            return "full"
        return "no-op"


@dataclass
class EWCOnlyStrategy:
    """Baseline C — EWC-regularized fine-tune of ALL layers on PSI alert.

    This isolates the value of EWC-as-regularization from CDAG-based targeting.
    """

    name: str = "ewc_only"
    cooldown_windows: int = 1
    _last_retrain: int = -999

    def decide(
        self,
        *,
        window_idx: int,
        adapter: ModelAdapter,
        window_X: np.ndarray,
        window_y: np.ndarray,
        trigger: PSITrigger,
        rolling_f1: float,
    ) -> str:
        alert = trigger.evaluate(window_X)
        if alert.fired and (window_idx - self._last_retrain) >= self.cooldown_windows:
            self._last_retrain = window_idx
            return "partial_all_layers"
        return "no-op"


# Type alias for imports from harness.py
Strategy = PSIFullRetrainStrategy | FixedScheduleStrategy | EWCOnlyStrategy
