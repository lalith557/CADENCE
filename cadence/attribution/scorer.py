"""CDAG-based responsibility scorer.

Drop-in replacement for `cadence.rso.scorers.PSIResponsibilityScorer` — the RSO
does not need to be retrained when we swap this in (that's the point of
decoupling Claim C from Claim A/B).

For Phase 3 (before the surrogate lands in Phase 4), the responsibility of a
feature node is scored by:
  1. Build a CDAG from the current window + a reference baseline stats.
  2. For each feature node f, propagate its edge weights through the DAG until
     the performance node is reached.
  3. Score(f) = |sum of directed path weights f → ... → performance|,
     normalized to [0, 1] by softmax.

This is a "structural attribution" proxy — no learned surrogate yet. In Phase 4
the surrogate replaces this ad-hoc propagation with a learned counterfactual
prediction. But even this proxy should beat PSI in H1: it uses the CDAG's
knowledge of *which cluster* / *which layer* the drifted feature actually
propagates through, whereas PSI is blind to internal structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cadence.adapters.base import ModelAdapter
from cadence.cdag import (
    ActivationTap,
    CDAGNodeSet,
    build_baseline_stats,
    build_cdag_from_windows,
    compute_windowed_signals,
)


@dataclass
class CDAGResponsibilityScorer:
    """Score = normalized directed-path strength from each feature node to
    the performance node in the current window's CDAG."""

    tap: ActivationTap
    node_set: CDAGNodeSet
    baseline_means: np.ndarray
    baseline_stds: np.ndarray
    window_size: int = 512
    max_path_length: int = 4
    pc_alpha: float = 0.05
    notears_lambda: float = 0.05
    notears_max_iter: int = 25
    name: str = "cdag_structural"

    @classmethod
    def build(
        cls,
        adapter: ModelAdapter,
        baseline_X: np.ndarray,
        feature_names: list[str],
        baseline_f1: float,
        *,
        k_overrides: dict[str, int] | None = None,
        window_size: int = 512,
    ) -> CDAGResponsibilityScorer:
        from cadence.cdag.nodes import build_node_set

        tap = ActivationTap(adapter).fit(baseline_X, k_overrides=k_overrides)
        node_set = build_node_set(feature_names, tap)
        bmean, bstd = build_baseline_stats(baseline_X, tap, node_set, baseline_f1)
        return cls(
            tap=tap,
            node_set=node_set,
            baseline_means=bmean,
            baseline_stds=bstd,
            window_size=window_size,
        )

    def __call__(
        self, adapter: ModelAdapter, window_X: np.ndarray, window_y: np.ndarray
    ) -> np.ndarray:
        # Estimate current F1 to feed the performance node.
        from sklearn.metrics import f1_score

        threshold = getattr(adapter, "decision_threshold", 0.5)
        probs = adapter.predict_proba(window_X)
        preds = (probs >= threshold).astype(np.int64)
        current_f1 = float(f1_score(window_y, preds, zero_division=0))

        # Attach the tap to the (possibly-updated) adapter — same layer topology.
        self.tap.adapter = adapter

        # Build the CDAG for this window.
        w_size = min(self.window_size, max(1, window_X.shape[0] // 4))
        signals = compute_windowed_signals(
            window_X,
            window_y,
            self.tap,
            self.node_set,
            self.baseline_means,
            self.baseline_stds,
            window_f1=current_f1,
            window_size=w_size,
        )
        cdag = build_cdag_from_windows(
            signals,
            pc_alpha=self.pc_alpha,
            notears_lambda=self.notears_lambda,
            notears_max_iter=self.notears_max_iter,
        )

        # Propagate feature-node influence to the performance node via truncated
        # path sum (matrix power).
        W = np.abs(cdag.weights).astype(np.float32)
        perf_idx = self.node_set.performance_index
        # influence[i] = Σ_{k=1..K} (W^k)[i, perf_idx]
        power = W.copy()
        influence = np.zeros(W.shape[0], dtype=np.float32)
        influence += power[:, perf_idx]
        for _ in range(self.max_path_length - 1):
            power = power @ W
            influence += power[:, perf_idx]

        # Restrict to feature nodes only and normalize.
        feat_scores = np.zeros(len(self.node_set.feature_indices), dtype=np.float32)
        for i, idx in enumerate(self.node_set.feature_indices):
            feat_scores[i] = influence[idx]

        # Fall back to drift-z magnitude if the CDAG produced ~zero influence
        # (happens on very short windows / when PC returns no skeleton).
        if float(feat_scores.sum()) < 1e-9:
            # last-window drift-z per feature node as fallback signal
            last_stats = signals[-1]  # (n_nodes, 4)
            for i, idx in enumerate(self.node_set.feature_indices):
                feat_scores[i] = abs(float(last_stats[idx, 3]))

        # Softmax-like normalization to [0, 1].
        z = feat_scores - feat_scores.max()
        exp_z = np.exp(z)
        out = exp_z / (exp_z.sum() + 1e-9)
        # Rescale so the max is 1.0 (matches PSI stub's saturating map).
        out = out / (out.max() + 1e-9)
        return out.astype(np.float32)
