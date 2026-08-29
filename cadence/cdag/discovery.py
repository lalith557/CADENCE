"""Hybrid causal discovery: PC (skeleton) + NOTEARS-style weighted refinement.

The reference brief calls for `causal-learn` PC + our own NOTEARS wrapper.
For Phase 3, we use:

1. **Skeleton via PC (causal-learn)** on windowed node time-series.
   Conditional-independence testing across the (n_windows, n_nodes) matrix of
   windowed drift-z summary stats. PC returns undirected + partial orientations
   over the n_nodes × n_nodes skeleton.

2. **Weighted DAG via a small NOTEARS-style continuous relaxation**, restricted
   to the PC-approved edges. Minimizes reconstruction MSE + acyclicity penalty
   + L1 sparsity, with all non-skeleton edges hard-pinned to zero.

The output is a real-valued weighted adjacency matrix `W ∈ R^{n×n}` where
`|W[i, j]|` = estimated causal strength of i → j.

Kept intentionally minimal: our own NOTEARS implementation is ~40 lines of
numpy so we don't take on gcastle's Windows install pain (see D-13 rationale
for the "avoid heavy deps" preference).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cadence.common.logging import get_logger

log = get_logger("cadence.cdag.discovery")


@dataclass
class CDAG:
    """Weighted DAG over the CDAG's nodes."""

    weights: np.ndarray  # (n, n); weights[i, j] = strength i -> j
    skeleton: np.ndarray  # (n, n) boolean — PC's skeleton mask


def _pc_skeleton(x: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Run causal-learn PC to get an undirected skeleton.

    `x` shape: (n_samples, n_features). Returns (n_features, n_features) bool
    mask. Falls back to a *dense* skeleton (all edges allowed) if causal-learn
    fails or n_samples is too small — we still want NOTEARS to produce SOMEthing.
    """
    n_features = x.shape[1]
    try:
        from causallearn.search.ConstraintBased.PC import pc as _pc

        # PC returns a CausalGraph; adjacency-matrix method varies by version.
        cg = _pc(x, alpha=alpha, indep_test="fisherz", show_progress=False, verbose=False)
        graph_matrix = cg.G.graph
        # causal-learn's `graph` encoding: 0 = no edge, non-zero = edge (any orientation).
        skel = (np.array(graph_matrix) != 0).astype(bool)
        # Symmetric OR — treat as undirected skeleton.
        skel = skel | skel.T
        np.fill_diagonal(skel, False)
        log.info("pc_skeleton_ok", n_edges=int(skel.sum() // 2), alpha=alpha)
        return skel
    except Exception as e:  # pragma: no cover — PC can be finicky on tiny samples
        log.warning("pc_failed_fallback_dense", err=str(e))
        skel = np.ones((n_features, n_features), dtype=bool)
        np.fill_diagonal(skel, False)
        return skel


def _notears_masked(
    x: np.ndarray,
    skeleton: np.ndarray,
    *,
    lambda_l1: float = 0.01,
    max_iter: int = 100,
    lr: float = 1e-2,
    rho_max: float = 1e6,
) -> np.ndarray:
    """Minimal masked NOTEARS. Restricts non-skeleton entries to zero.

    Objective (per NOTEARS): 0.5/n · ||X - X @ W||² + λ ||W||₁ + ρ/2 · h(W)²
    with h(W) = tr(exp(W ⊙ W)) - n_features (differentiable acyclicity).

    We keep this deliberately simple — augmented-Lagrangian outer loop, gradient
    inner loop, all in numpy. Good enough for the small graphs (≤ ~80 nodes).
    """
    n, d = x.shape
    mask = skeleton.astype(np.float32)
    # W is d×d; use a small init so h(W) starts small.
    W = np.zeros((d, d), dtype=np.float32)
    alpha, rho, h_prev = 0.0, 1.0, np.inf

    def _loss_and_grad(W):
        M = W * mask
        pred = x @ M
        resid = x - pred
        loss = 0.5 / n * float(np.sum(resid**2))
        grad_recon = -1.0 / n * (x.T @ resid) * mask
        # Acyclicity: h(W) = tr(exp(M∘M)) - d
        W2 = M * M
        expm = np.linalg.matrix_power(np.eye(d) + W2 / d, d)  # cheap approx
        h = float(np.trace(expm)) - d
        grad_h = 2.0 * M  # gradient of tr((I + W²/d)^d) — approximation
        grad_h = grad_h * mask
        return loss, grad_recon, h, grad_h

    for outer in range(20):
        for _ in range(max_iter):
            loss, grad_recon, h, grad_h = _loss_and_grad(W)
            # L1 subgradient — soft-threshold-ish
            grad_l1 = lambda_l1 * np.sign(W)
            grad = grad_recon + grad_l1 + (rho * h + alpha) * grad_h
            W = W - lr * grad
            W = W * mask  # re-mask each step
        loss, _, h, _ = _loss_and_grad(W)
        log.info("notears_outer", outer=outer, loss=loss, h=h, rho=rho)
        if h < 1e-8 or rho > rho_max:
            break
        if h > 0.25 * h_prev:
            rho *= 10.0
        alpha += rho * h
        h_prev = h

    return W


def build_cdag_from_windows(
    windowed_signals: np.ndarray,
    *,
    stat_index: int = 3,  # 3 = drift_z (from cadence.cdag.nodes._summary_stats)
    pc_alpha: float = 0.05,
    notears_lambda: float = 0.01,
    notears_max_iter: int = 50,
) -> CDAG:
    """Build a CDAG from `(n_windows, n_nodes, 4)` per-node stats.

    Uses one selected stat (default: drift_z) as the per-node time-series that
    PC and NOTEARS operate on.
    """
    x = windowed_signals[:, :, stat_index]  # (n_windows, n_nodes)
    if x.shape[0] < 3:
        # Not enough windows for PC — return dense skeleton, zero weights.
        d = x.shape[1]
        skel = np.ones((d, d), dtype=bool)
        np.fill_diagonal(skel, False)
        return CDAG(weights=np.zeros((d, d), dtype=np.float32), skeleton=skel)

    # Center each column so PC's Fisher-z test is on standardized data.
    xc = x - x.mean(axis=0, keepdims=True)
    xc = xc / (x.std(axis=0, keepdims=True) + 1e-6)

    skeleton = _pc_skeleton(xc, alpha=pc_alpha)
    W = _notears_masked(xc, skeleton, lambda_l1=notears_lambda, max_iter=notears_max_iter)
    return CDAG(weights=W, skeleton=skeleton)
