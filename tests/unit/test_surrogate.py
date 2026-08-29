"""Guards for the Step C surrogate + joint training."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cadence.attribution.gnn import CDAGSage, GNNConfig, cdag_to_pyg_data
from cadence.attribution.sandbox_labels import InterventionSample
from cadence.attribution.surrogate import (
    JointTrainConfig,
    SurrogateConfig,
    SurrogateMLP,
    build_surrogate_on_device,
    train_joint,
)


def _fake_intervention_samples(
    n: int, n_nodes: int = 5, device: torch.device | None = None
) -> list[InterventionSample]:
    rng = np.random.default_rng(0)
    dev = device or torch.device("cpu")
    W = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    W[0, 1] = 0.5
    W[1, 2] = 0.7
    samples = []
    for i in range(n):
        X = rng.standard_normal((n_nodes, 4)).astype(np.float32)
        data = cdag_to_pyg_data(W, X, device=dev)
        # Fake targets that correlate with candidate node's drift_z sign.
        drift_z = float(X[i % n_nodes, 3])
        pre = 0.75
        post = float(np.clip(pre + 0.1 * drift_z, 0.0, 1.0))
        samples.append(
            InterventionSample(
                data=data,
                candidate_feature_idx=i % n_nodes,
                candidate_node_idx=i % n_nodes,
                root_cause_feature_idx=0,
                pre_fix_f1=pre,
                post_fix_f1=post,
                severity=float(rng.uniform(0.2, 1.5)),
                mechanism="covariate_shift",
            )
        )
    return samples


def test_surrogate_forward_produces_sigmoid_output_shape() -> None:
    m = SurrogateMLP(SurrogateConfig(embedding_dim=16, context_dim=2, hidden_dim=8))
    m.eval()
    emb = torch.randn(4, 16)
    ctx = torch.randn(4, 2)
    out = m(emb, ctx)
    assert out.shape == (4,)
    assert torch.all(out >= 0) and torch.all(out <= 1)


@pytest.mark.gpu
def test_build_surrogate_on_device_places_params_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    m = build_surrogate_on_device(SurrogateConfig())
    for p in m.parameters():
        assert p.device.type == "cuda"


def test_train_joint_reduces_loss() -> None:
    """Joint training should measurably reduce MSE on a small synthetic set."""
    dev = torch.device("cpu")
    gnn = CDAGSage(GNNConfig(input_dim=4, hidden_dim=16, embedding_dim=8)).to(dev)
    surrogate = SurrogateMLP(SurrogateConfig(embedding_dim=8, hidden_dim=16)).to(dev)
    samples = _fake_intervention_samples(24, n_nodes=5, device=dev)
    summary = train_joint(
        gnn,
        surrogate,
        samples,
        cfg=JointTrainConfig(lr=1e-2, epochs=15, val_frac=0.25, seed=7),
        device=dev,
    )
    # First-epoch loss should exceed final-epoch loss.
    first_mse = summary["history"][0]["train_mse"]
    final_mse = summary["history"][-1]["train_mse"]
    assert final_mse < first_mse, f"loss did not decrease: {first_mse} -> {final_mse}"


@pytest.mark.gpu
def test_train_joint_logs_max_vram_on_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    from cadence.attribution.gnn import build_model_on_device

    gnn = build_model_on_device(GNNConfig(embedding_dim=16))
    surrogate = build_surrogate_on_device(SurrogateConfig(embedding_dim=16))
    dev = torch.device("cuda")
    samples = _fake_intervention_samples(16, n_nodes=5, device=dev)
    summary = train_joint(
        gnn,
        surrogate,
        samples,
        cfg=JointTrainConfig(epochs=3, val_frac=0.25),
        device=dev,
    )
    assert "max_vram_mb" in summary
    assert summary["max_vram_mb"] > 0
