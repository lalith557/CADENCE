"""Guards for the Step B GNN encoder + sandbox-labels pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cadence.attribution.gnn import (
    CDAGSage,
    GNNConfig,
    build_model_on_device,
    cdag_to_pyg_data,
    encode_cdag,
)


def _small_dag_weights() -> np.ndarray:
    # 5 nodes, deliberately sparse.
    W = np.zeros((5, 5), dtype=np.float32)
    W[0, 1] = 0.8
    W[1, 2] = 0.6
    W[2, 3] = 0.5
    W[3, 4] = 0.7
    return W


def test_cdag_to_pyg_data_builds_expected_edges() -> None:
    W = _small_dag_weights()
    X = np.random.RandomState(0).standard_normal((5, 4)).astype(np.float32)
    data = cdag_to_pyg_data(W, X, device=torch.device("cpu"))
    assert data.edge_index.shape[1] == 4  # 4 non-zero directed edges
    assert data.edge_weight.shape[0] == 4
    assert torch.all(data.edge_weight > 0)


def test_cdag_to_pyg_data_falls_back_to_self_loops_on_empty_edges() -> None:
    W = np.zeros((5, 5), dtype=np.float32)
    X = np.zeros((5, 4), dtype=np.float32)
    data = cdag_to_pyg_data(W, X, device=torch.device("cpu"))
    # Fallback: 5 self-loops so PyG doesn't crash on empty edge_index.
    assert data.edge_index.shape[1] == 5


def test_gnn_forward_produces_expected_shapes() -> None:
    model = CDAGSage(GNNConfig(input_dim=4, hidden_dim=16, embedding_dim=8))
    model.eval()
    W = _small_dag_weights()
    X = np.random.RandomState(0).standard_normal((5, 4)).astype(np.float32)
    logits, emb = encode_cdag(model, W, X, device=torch.device("cpu"))
    assert logits.shape == (5,)
    assert emb.shape == (5, 8)


@pytest.mark.gpu
def test_build_model_on_device_uses_cuda_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    model = build_model_on_device(GNNConfig())
    for p in model.parameters():
        assert p.device.type == "cuda"


@pytest.mark.gpu
def test_encode_cdag_on_gpu_produces_finite_values() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    model = build_model_on_device(GNNConfig())
    W = _small_dag_weights()
    X = np.random.RandomState(1).standard_normal((5, 4)).astype(np.float32)
    logits, emb = encode_cdag(model, W, X)  # device auto-resolves to cuda
    assert np.isfinite(logits).all()
    assert np.isfinite(emb).all()
