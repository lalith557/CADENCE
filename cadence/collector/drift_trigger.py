"""Drift-detection trigger — PSI per feature + delayed-label F1 (D-10).

This is the FAIR baseline trigger that CADENCE and all its comparison baselines
share, so CADENCE's causal-attribution value-add is not confounded by having
better detection than the baselines.

PSI = Σ (p_i - q_i) * ln(p_i / q_i)  over histogram bins.
Standard thresholds: <0.1 stable, 0.1-0.25 moderate, >0.25 significant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score


@dataclass
class TriggerAlert:
    fired: bool
    reason: Literal["psi", "f1", "ks", "none"]
    feature_idx: int | None
    feature_name: str | None
    psi_value: float | None
    ks_pvalue: float | None
    delayed_f1: float | None
    n_baseline: int
    n_window: int


@dataclass
class DriftTriggerConfig:
    psi_threshold: float = 0.25
    ks_alpha: float = 0.01
    f1_sla: float | None = None  # if not None, triggers when rolling F1 < sla
    n_bins: int = 10
    min_window_rows: int = 200


class PSITrigger:
    """Population Stability Index over a rolling window, per feature.

    Uses training-distribution deciles as bin edges (fit once at construction).
    """

    def __init__(
        self,
        baseline_X: np.ndarray,
        feature_names: list[str],
        cfg: DriftTriggerConfig | None = None,
    ):
        self.baseline_X = baseline_X
        self.feature_names = feature_names
        self.cfg = cfg or DriftTriggerConfig()
        self._bin_edges = self._fit_bin_edges(baseline_X, self.cfg.n_bins)
        self._baseline_hist = self._histograms(baseline_X, self._bin_edges)

    @staticmethod
    def _fit_bin_edges(X: np.ndarray, n_bins: int) -> list[np.ndarray]:
        edges = []
        for j in range(X.shape[1]):
            col = X[:, j]
            qs = np.quantile(col, np.linspace(0, 1, n_bins + 1))
            qs[0] = -np.inf
            qs[-1] = np.inf
            # Ensure strictly increasing; jitter dupes upward.
            for i in range(1, len(qs)):
                if qs[i] <= qs[i - 1]:
                    qs[i] = qs[i - 1] + 1e-9
            edges.append(qs)
        return edges

    @staticmethod
    def _histograms(X: np.ndarray, edges: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((X.shape[1], len(edges[0]) - 1), dtype=np.float64)
        for j in range(X.shape[1]):
            hist, _ = np.histogram(X[:, j], bins=edges[j])
            total = hist.sum()
            if total > 0:
                out[j] = hist / total
        return out

    def psi_per_feature(self, window_X: np.ndarray) -> np.ndarray:
        """Return per-feature PSI values [n_features]."""
        window_hist = self._histograms(window_X, self._bin_edges)
        eps = 1e-6
        p = window_hist + eps
        q = self._baseline_hist + eps
        # Normalize after epsilon add so the sum is still 1.
        p = p / p.sum(axis=1, keepdims=True)
        q = q / q.sum(axis=1, keepdims=True)
        psi = ((p - q) * np.log(p / q)).sum(axis=1)
        return psi.astype(np.float32)

    def evaluate(
        self,
        window_X: np.ndarray,
        *,
        delayed_labels: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> TriggerAlert:
        """Run PSI + optional F1 check. Returns the strongest firing reason."""
        n_baseline = self.baseline_X.shape[0]
        n_window = window_X.shape[0]

        if n_window < self.cfg.min_window_rows:
            return TriggerAlert(
                fired=False,
                reason="none",
                feature_idx=None,
                feature_name=None,
                psi_value=None,
                ks_pvalue=None,
                delayed_f1=None,
                n_baseline=n_baseline,
                n_window=n_window,
            )

        psi = self.psi_per_feature(window_X)
        worst = int(np.argmax(psi))
        worst_psi = float(psi[worst])

        delayed_f1_val: float | None = None
        if delayed_labels is not None:
            y_true, y_pred_proba = delayed_labels
            if y_true.size > 0:
                delayed_f1_val = float(
                    f1_score(y_true, (y_pred_proba >= 0.5).astype(np.int64), zero_division=0)
                )

        # Reason ranking: F1 breach > PSI breach > KS breach.
        if (
            self.cfg.f1_sla is not None
            and delayed_f1_val is not None
            and delayed_f1_val < self.cfg.f1_sla
        ):
            return TriggerAlert(
                fired=True,
                reason="f1",
                feature_idx=worst if worst_psi > self.cfg.psi_threshold else None,
                feature_name=self.feature_names[worst]
                if worst_psi > self.cfg.psi_threshold
                else None,
                psi_value=worst_psi,
                ks_pvalue=None,
                delayed_f1=delayed_f1_val,
                n_baseline=n_baseline,
                n_window=n_window,
            )

        if worst_psi > self.cfg.psi_threshold:
            return TriggerAlert(
                fired=True,
                reason="psi",
                feature_idx=worst,
                feature_name=self.feature_names[worst],
                psi_value=worst_psi,
                ks_pvalue=None,
                delayed_f1=delayed_f1_val,
                n_baseline=n_baseline,
                n_window=n_window,
            )

        return TriggerAlert(
            fired=False,
            reason="none",
            feature_idx=worst,
            feature_name=self.feature_names[worst],
            psi_value=worst_psi,
            ks_pvalue=None,
            delayed_f1=delayed_f1_val,
            n_baseline=n_baseline,
            n_window=n_window,
        )


def ks_test_per_feature(baseline_X: np.ndarray, window_X: np.ndarray) -> np.ndarray:
    """Two-sample KS test per feature — used as a secondary detector / ablation baseline."""
    pvals = np.empty(baseline_X.shape[1], dtype=np.float32)
    for j in range(baseline_X.shape[1]):
        try:
            _, p = stats.ks_2samp(baseline_X[:, j], window_X[:, j])
        except ValueError:
            p = 1.0
        pvals[j] = p
    return pvals
