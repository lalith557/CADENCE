"""Counterfactual surrogate — predicts post-fix F1 from GNN embeddings.

Fills in Table 3D of the reference:

  Input  : GNN(node_embedding) + context (pre_fix_f1, severity)
  Output : predicted post-fix F1 (scalar)
  Loss   : MSE against measured post-fix F1 from sandbox interventions
  Training: joint with the GNN — gradients flow through both, per Q24-Q26.

At inference time the surrogate is used to compute responsibility scores:

    responsibility(feature) = predicted_post_fix_f1_if_fixed - pre_fix_f1

Higher score = fixing this feature is predicted to help more.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from cadence.attribution.gnn import CDAGSage, GNNConfig, build_model_on_device
from cadence.attribution.sandbox_labels import InterventionSample
from cadence.common.device import (
    assert_on_gpu,
    cuda_memory_snapshot,
    get_device,
    log_device_info,
    reset_peak_memory,
)
from cadence.common.logging import get_logger

log = get_logger("cadence.attribution.surrogate")


@dataclass
class SurrogateConfig:
    embedding_dim: int = 16  # must match GNNConfig.embedding_dim
    context_dim: int = 2  # (pre_fix_f1, severity)
    hidden_dim: int = 32
    dropout: float = 0.1
    device: str = "auto"


class SurrogateMLP(nn.Module):
    """Small MLP that maps (node embedding + context) -> predicted post-fix F1.

    Output is passed through sigmoid so predictions live in [0, 1] — F1 range.
    """

    def __init__(self, cfg: SurrogateConfig | None = None):
        super().__init__()
        self.cfg = cfg or SurrogateConfig()
        in_dim = self.cfg.embedding_dim + self.cfg.context_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, 1),
        )

    def forward(self, node_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """node_emb: (B, embedding_dim); context: (B, context_dim). Returns (B,)."""
        x = torch.cat([node_emb, context], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)


def build_surrogate_on_device(cfg: SurrogateConfig | None = None) -> SurrogateMLP:
    cfg = cfg or SurrogateConfig()
    m = SurrogateMLP(cfg)
    dev = get_device(override=cfg.device if cfg.device != "auto" else None)
    m = m.to(dev)
    log_device_info(dev)
    if dev.type == "cuda":
        assert_on_gpu(m, name="SurrogateMLP")
    return m


@dataclass
class JointTrainConfig:
    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    val_frac: float = 0.2
    seed: int = 42


def train_joint(
    gnn: CDAGSage,
    surrogate: SurrogateMLP,
    samples: list[InterventionSample],
    cfg: JointTrainConfig | None = None,
    *,
    device: torch.device | None = None,
) -> dict:
    """Joint MSE training of the GNN + surrogate.

    For each intervention triple:
      * run the GNN on the sample's CDAG
      * pick the candidate node's embedding
      * build the context vector [pre_fix_f1, severity]
      * predict post_fix_f1 through the surrogate
      * MSE against measured post_fix_f1

    Both networks are on cuda; gradient flows through both.
    """
    cfg = cfg or JointTrainConfig()
    dev = device if device is not None else get_device()

    if dev.type == "cuda":
        reset_peak_memory()
        log_device_info(dev)
        assert_on_gpu(gnn, name="CDAGSage[joint]")
        assert_on_gpu(surrogate, name="SurrogateMLP[joint]")

    rng = np.random.default_rng(cfg.seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_val = max(1, int(cfg.val_frac * len(idx)))
    val_idx = [int(i) for i in idx[:n_val]]
    train_idx = [int(i) for i in idx[n_val:]]

    params = list(gnn.parameters()) + list(surrogate.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: list[dict] = []

    def _epoch(loader: list[int], train: bool) -> tuple[float, float, list, list]:
        gnn.train(mode=train)
        surrogate.train(mode=train)
        total_loss = 0.0
        total_n = 0
        preds_all: list[float] = []
        targets_all: list[float] = []
        for i in loader:
            s = samples[i]
            data: Data = s.data
            if train:
                opt.zero_grad(set_to_none=True)
            _, emb = gnn(data)
            node_emb = emb[s.candidate_node_idx : s.candidate_node_idx + 1]  # (1, emb_dim)
            ctx = torch.tensor(
                [[float(s.pre_fix_f1), float(s.severity)]],
                dtype=torch.float32,
                device=dev,
            )
            pred = surrogate(node_emb, ctx)
            target = torch.tensor([float(s.post_fix_f1)], dtype=torch.float32, device=dev)
            loss = F.mse_loss(pred, target)
            if train:
                loss.backward()
                opt.step()
            total_loss += float(loss.item())
            total_n += 1
            preds_all.append(float(pred.item()))
            targets_all.append(float(s.post_fix_f1))
        return total_loss / max(total_n, 1), 0.0, preds_all, targets_all

    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        train_loss, _, _, _ = _epoch(train_idx, train=True)
        val_loss, _, val_preds, val_targets = _epoch(val_idx, train=False)
        mae = float(np.mean(np.abs(np.array(val_preds) - np.array(val_targets))))
        r2 = _r2(val_preds, val_targets)
        entry = {
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val_loss,
            "val_mae": mae,
            "val_r2": r2,
        }
        history.append(entry)
        log.info("surrogate_epoch", **entry)

    final_val_loss, _, val_preds, val_targets = _epoch(val_idx, train=False)
    summary = {
        "wall_s": time.perf_counter() - t0,
        "final_val_mse": final_val_loss,
        "final_val_mae": float(np.mean(np.abs(np.array(val_preds) - np.array(val_targets)))),
        "final_val_r2": _r2(val_preds, val_targets),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "val_preds": val_preds,
        "val_targets": val_targets,
        "history": history,
    }
    if dev.type == "cuda":
        snap = cuda_memory_snapshot()
        summary["max_vram_mb"] = round(snap["max_allocated"] / 1024**2, 3)
        summary["allocated_vram_mb"] = round(snap["allocated"] / 1024**2, 3)
        log.info(
            "surrogate_train_done",
            wall_s=summary["wall_s"],
            final_val_mae=summary["final_val_mae"],
            final_val_r2=summary["final_val_r2"],
            max_vram_mb=summary["max_vram_mb"],
        )
    else:
        log.info(
            "surrogate_train_done",
            wall_s=summary["wall_s"],
            final_val_mae=summary["final_val_mae"],
            final_val_r2=summary["final_val_r2"],
        )
    return summary


def _r2(preds: list[float], targets: list[float]) -> float:
    p = np.array(preds, dtype=np.float64)
    t = np.array(targets, dtype=np.float64)
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot
