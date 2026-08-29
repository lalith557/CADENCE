"""CDAG node construction (D-3).

Node types:
  * external feature nodes (one per input feature)
  * activation-cluster nodes (one per k-means cluster per tapped layer)
  * one outcome / performance node

Each node has a *per-window feature vector* (mean, var, skew, drift-magnitude)
that feeds both the causal discovery step (windowed samples for PC/NOTEARS) and
the GNN's per-node input features (Table 3B in the reference).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats as sci_stats

from cadence.cdag.tap import ActivationTap

NodeKind = Literal["feature", "cluster", "performance"]


@dataclass(frozen=True)
class NodeSpec:
    idx: int
    kind: NodeKind
    name: str
    # For cluster nodes: which layer it lives in, and which centroid index.
    layer_name: str | None = None
    cluster_id: int | None = None


@dataclass
class CDAGNodeSet:
    specs: list[NodeSpec]
    feature_indices: list[int]
    cluster_indices: list[int]
    performance_index: int
    feature_names: list[str]

    @property
    def n_nodes(self) -> int:
        return len(self.specs)


def build_node_set(feature_names: list[str], tap: ActivationTap) -> CDAGNodeSet:
    """Deterministic index assignment: features first, then clusters (by layer
    order), then the performance node last."""
    specs: list[NodeSpec] = []
    feat_idx: list[int] = []
    clus_idx: list[int] = []

    for f_name in feature_names:
        specs.append(NodeSpec(idx=len(specs), kind="feature", name=f_name))
        feat_idx.append(specs[-1].idx)

    for layer_name, fit in tap.fits.items():
        for c in range(fit.k):
            specs.append(
                NodeSpec(
                    idx=len(specs),
                    kind="cluster",
                    name=f"{layer_name}_c{c}",
                    layer_name=layer_name,
                    cluster_id=c,
                )
            )
            clus_idx.append(specs[-1].idx)

    perf_idx = len(specs)
    specs.append(NodeSpec(idx=perf_idx, kind="performance", name="performance"))

    return CDAGNodeSet(
        specs=specs,
        feature_indices=feat_idx,
        cluster_indices=clus_idx,
        performance_index=perf_idx,
        feature_names=feature_names,
    )


def _summary_stats(x: np.ndarray, baseline_mean: float, baseline_std: float) -> np.ndarray:
    """(mean, var, skew, drift_z) — a 4-vector per node per window."""
    if x.size == 0:
        return np.zeros(4, dtype=np.float32)
    mean = float(np.mean(x))
    var = float(np.var(x))
    skew = float(sci_stats.skew(x, bias=False)) if x.size >= 3 else 0.0
    drift_z = (mean - baseline_mean) / (baseline_std + 1e-6)
    return np.array([mean, var, skew, drift_z], dtype=np.float32)


def build_baseline_stats(
    baseline_X: np.ndarray,
    tap: ActivationTap,
    node_set: CDAGNodeSet,
    baseline_f1: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-node (baseline_mean, baseline_std) for later drift-z scoring.

    Returns two arrays of shape (n_nodes,).
    """
    means = np.zeros(node_set.n_nodes, dtype=np.float32)
    stds = np.zeros(node_set.n_nodes, dtype=np.float32)
    # Feature nodes.
    for i, idx in enumerate(node_set.feature_indices):
        col = baseline_X[:, i]
        means[idx] = float(np.mean(col))
        stds[idx] = float(np.std(col) + 1e-6)
    # Cluster nodes.
    signals = tap.cluster_signals(baseline_X)
    for spec in node_set.specs:
        if spec.kind != "cluster":
            continue
        m = signals[spec.layer_name][:, spec.cluster_id]
        means[spec.idx] = float(np.mean(m))
        stds[spec.idx] = float(np.std(m) + 1e-6)
    # Performance node.
    means[node_set.performance_index] = float(baseline_f1)
    stds[node_set.performance_index] = 1e-6
    return means, stds


def compute_windowed_signals(
    X: np.ndarray,
    y: np.ndarray | None,
    tap: ActivationTap,
    node_set: CDAGNodeSet,
    baseline_means: np.ndarray,
    baseline_stds: np.ndarray,
    window_f1: float,
    window_size: int,
) -> np.ndarray:
    """Split X into `n_windows` chunks and compute the 4-stat vector per node
    per window. Returns array of shape (n_windows, n_nodes, 4).
    """
    n = X.shape[0]
    n_windows = max(1, n // window_size)
    out = np.zeros((n_windows, node_set.n_nodes, 4), dtype=np.float32)
    signals = tap.cluster_signals(X)

    for w in range(n_windows):
        s = w * window_size
        e = min(s + window_size, n)
        window_X = X[s:e]
        # Feature nodes.
        for i, idx in enumerate(node_set.feature_indices):
            out[w, idx] = _summary_stats(window_X[:, i], baseline_means[idx], baseline_stds[idx])
        # Cluster nodes.
        for spec in node_set.specs:
            if spec.kind != "cluster":
                continue
            col = signals[spec.layer_name][s:e, spec.cluster_id]
            out[w, spec.idx] = _summary_stats(
                col, baseline_means[spec.idx], baseline_stds[spec.idx]
            )
        # Performance node — same F1 across all windows in this pass.
        out[w, node_set.performance_index] = np.array(
            [
                window_f1,
                0.0,
                0.0,
                (window_f1 - baseline_means[node_set.performance_index])
                / (baseline_stds[node_set.performance_index] + 1e-6),
            ],
            dtype=np.float32,
        )
    return out
