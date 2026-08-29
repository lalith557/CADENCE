"""Activation tapping + per-layer k-means clustering.

For a given `ModelAdapter` and a slice of *baseline* (pre-drift) data:
1. Run the adapter's `forward_with_traces` to get post-ReLU activations per
   tapped layer.
2. Fit one k-means per layer on those activations (D-2: elbow / silhouette
   sweep from CDAG config's `cluster_k_sweep`, or a fixed override).
3. Store the fitted models; later, at CDAG-build time, we assign each row of a
   new (drifted) window to its nearest cluster and use the *distance to the
   pre-drift centroid* as that cluster's per-row activity signal.

The output of this file is what turns raw activations into the fixed set of
"activation-cluster nodes" the CDAG has.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from cadence.adapters.base import ModelAdapter
from cadence.common.logging import get_logger

log = get_logger("cadence.cdag.tap")


@dataclass
class ClusterFit:
    """Fitted k-means for one tapped layer."""

    layer_name: str
    k: int
    kmeans: KMeans
    centroid_norm: np.ndarray  # ||centroid|| per cluster, used as scale reference
    baseline_within_variance: float


@dataclass
class ActivationTap:
    adapter: ModelAdapter
    fits: dict[str, ClusterFit] = field(default_factory=dict)

    def fit(
        self,
        baseline_X: np.ndarray,
        *,
        k_sweep: tuple[int, int] = (2, 16),
        k_overrides: dict[str, int] | None = None,
        max_baseline_rows: int = 20_000,
        random_state: int = 0,
    ) -> ActivationTap:
        """Fit one k-means per tapped layer using the baseline activations.

        `k_overrides` maps layer name → fixed k (skips silhouette sweep, matches
        D-2's illustrative Layer1=8/Layer2=4 defaults when passed).
        """
        k_overrides = k_overrides or {}
        if baseline_X.shape[0] > max_baseline_rows:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(baseline_X.shape[0], size=max_baseline_rows, replace=False)
            baseline_X = baseline_X[idx]

        _, traces = self.adapter.forward_with_traces(baseline_X)

        for layer_name, activations in traces.items():
            if layer_name in k_overrides:
                k = k_overrides[layer_name]
            else:
                k = _pick_k_by_silhouette(
                    activations, k_min=k_sweep[0], k_max=k_sweep[1], random_state=random_state
                )
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            km.fit(activations)
            labels = km.labels_
            within_var = float(
                np.mean(np.linalg.norm(activations - km.cluster_centers_[labels], axis=1) ** 2)
            )
            centroid_norm = np.linalg.norm(km.cluster_centers_, axis=1)
            self.fits[layer_name] = ClusterFit(
                layer_name=layer_name,
                k=k,
                kmeans=km,
                centroid_norm=centroid_norm,
                baseline_within_variance=within_var,
            )
            log.info(
                "layer_clustered",
                layer=layer_name,
                k=k,
                within_var=within_var,
                n_rows=activations.shape[0],
            )
        return self

    def cluster_signals(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """For each layer, return a (n_rows, k) matrix of *distance to each
        cluster centroid* — used as the per-cluster activation signal at CDAG
        build time. Bigger distance = row's activation drifted further from
        the corresponding pre-drift cluster.
        """
        if not self.fits:
            raise RuntimeError("ActivationTap.cluster_signals called before fit()")
        _, traces = self.adapter.forward_with_traces(X)
        out: dict[str, np.ndarray] = {}
        for layer_name, activations in traces.items():
            fit = self.fits.get(layer_name)
            if fit is None:
                continue
            # sklearn KMeans.transform returns Euclidean distance to each centroid.
            distances = fit.kmeans.transform(activations).astype(np.float32)
            out[layer_name] = distances
        return out


def _pick_k_by_silhouette(
    activations: np.ndarray,
    *,
    k_min: int,
    k_max: int,
    random_state: int,
    max_rows_for_silhouette: int = 5000,
) -> int:
    """Silhouette-score sweep with a small hardcoded max on k to keep it fast."""
    n_rows, n_dim = activations.shape
    k_max = min(k_max, max(k_min + 1, n_dim // 4 or k_min + 1))
    if n_rows > max_rows_for_silhouette:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_rows, size=max_rows_for_silhouette, replace=False)
        activations = activations[idx]

    best_k = k_min
    best_score = -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=5, random_state=random_state)
        labels = km.fit_predict(activations)
        if len(set(labels)) < 2:
            continue
        try:
            s = silhouette_score(activations, labels, sample_size=1000, random_state=random_state)
        except ValueError:
            continue
        if s > best_score:
            best_score = float(s)
            best_k = k
    return best_k
