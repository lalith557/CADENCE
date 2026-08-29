"""Responsibility scorers.

The sandbox env expects a `ResponsibilityScorer` callable: given an adapter and
a window of data, produce a per-feature score in [0, 1] indicating "how
responsible this feature is for the current performance drop."

Phase 2 uses the PSI-based stub. Phase 3 replaces it with the CDAG-GNN-surrogate
pipeline; the scorer interface is unchanged so PPO does not need to be
retrained just because the scorer becomes more sophisticated (this is on
purpose — it decouples Claim C from Claim A/B).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from cadence.adapters.base import ModelAdapter
from cadence.collector.drift_trigger import PSITrigger


class ResponsibilityScorer(Protocol):
    """Protocol every scorer implements."""

    name: str

    def __call__(
        self, adapter: ModelAdapter, window_X: np.ndarray, window_y: np.ndarray
    ) -> np.ndarray:
        """Return per-feature responsibility scores in [0, 1], length = n_features."""
        ...


@dataclass
class PSIResponsibilityScorer:
    """Phase 2 stub — PSI itself acts as responsibility.

    Rationale: without CDAG we have no learned notion of "which node caused it,"
    so the least biased fallback is to report the same statistical signal the
    trigger already computes. This is what the ablation "no GNN, no surrogate,
    use PSI directly" means in the paper.

    Bounded to [0, 1] by a saturating map: score = 1 - exp(-PSI / scale), which
    is smooth and matches PSI's convention (>0.25 = large, >0.5 = severe).
    """

    trigger: PSITrigger
    name: str = "psi_stub"
    saturation_scale: float = 0.3

    def __call__(
        self, adapter: ModelAdapter, window_X: np.ndarray, window_y: np.ndarray
    ) -> np.ndarray:
        psi = self.trigger.psi_per_feature(window_X)
        return 1.0 - np.exp(-psi / self.saturation_scale)
