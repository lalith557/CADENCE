"""GNN encoder over the CDAG, on GPU (Execution Plan §1b.2 + Step B).

The reference calls for "GraphSAGE/GAT-style" (Table 3C, Part 3). We use
GCNConv rather than SAGEConv because it accepts the |NOTEARS weight| as
`edge_weight` out of the box; SAGEConv in PyG 2.7 does not. The functional
behaviour is close enough — both are message-passing, mean-aggregated
convolutions over local neighbourhoods — and switching to GATv2Conv is a
one-line ablation once Step C's surrogate lands.

The CDAG is small (30 feature + ~12 cluster + 1 performance = O(40)) and
each edge has a learned NOTEARS weight, so a lightweight 2-layer conv is
plenty:

  Input per node : 4-vector (mean, var, skew, drift_z)
  Edge weight    : |NOTEARS weight| (fed to GCNConv as edge_weight)
  Encoder        : GCNConv(4 -> 32, relu) -> GCNConv(32 -> 16)  (2 layers)
  Readout        : Linear(16 -> 1) per node, softmax over feature nodes only.

The point of the softmax is that the responsibility scores form a probability
distribution — matches the reference's "responsibility score = predicted
recovery share." This is a pre-surrogate approximation; Step C will replace
the readout with the joint (GNN + surrogate) counterfactual head.

Training loop lives in `cadence.attribution.gnn_scorer` — this module owns
only the architecture + a pure-inference `encode()` helper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from cadence.common.device import assert_on_gpu, get_device, log_device_info


@dataclass
class GNNConfig:
    input_dim: int = 4  # mean, var, skew, drift_z
    hidden_dim: int = 32
    embedding_dim: int = 16
    dropout: float = 0.1
    device: str = "auto"


class CDAGSage(nn.Module):
    """2-layer GCN over the CDAG's directed weighted edges.

    Note on edges: NOTEARS returns a signed weighted adjacency W[i, j] =
    strength of i -> j. We feed |W[i, j]| as `edge_weight`; direction is
    preserved via the (source, target) pairs from the DIRECTED edge set.

    Named CDAGSage for continuity with the reference's "GraphSAGE/GAT-style"
    language — internally it's GCNConv, which is the closest weighted-conv
    match in PyG 2.7 that accepts edge_weight without extra plumbing.
    """

    def __init__(self, cfg: GNNConfig | None = None):
        super().__init__()
        self.cfg = cfg or GNNConfig()
        self.conv1 = GCNConv(self.cfg.input_dim, self.cfg.hidden_dim, add_self_loops=True)
        self.conv2 = GCNConv(self.cfg.hidden_dim, self.cfg.embedding_dim, add_self_loops=True)
        self.dropout = nn.Dropout(self.cfg.dropout)
        # Read-out head: one scalar per node. The scorer applies softmax over
        # feature nodes only.
        self.readout = nn.Linear(self.cfg.embedding_dim, 1)

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (per-node logits, per-node embeddings). Both live on `data.x.device`."""
        h = data.x
        edge_weight = getattr(data, "edge_weight", None)
        h = F.relu(self.conv1(h, data.edge_index, edge_weight=edge_weight))
        h = self.dropout(h)
        h = self.conv2(h, data.edge_index, edge_weight=edge_weight)
        logits = self.readout(h).squeeze(-1)
        return logits, h


def cdag_to_pyg_data(
    weights: np.ndarray,
    node_features: np.ndarray,
    *,
    edge_weight_floor: float = 1e-4,
    device: torch.device | None = None,
) -> Data:
    """Convert a NOTEARS-weighted CDAG + per-node feature matrix into a PyG
    `Data` object.

    `weights` : (n_nodes, n_nodes) — NOTEARS output, `weights[i, j] = i -> j`.
    `node_features` : (n_nodes, input_dim) — per-node stats.

    Edges with |weight| < `edge_weight_floor` are pruned so the message-
    passing budget isn't wasted on the DAG's noise floor.
    """
    dev = device if device is not None else get_device()
    W = np.asarray(weights, dtype=np.float32)
    mask = np.abs(W) >= edge_weight_floor
    np.fill_diagonal(mask, False)
    srcs, dsts = np.where(mask)
    if srcs.size == 0:
        # Every node isolated: add self-loops so message passing degenerates
        # gracefully instead of crashing PyG on an empty edge_index.
        n = W.shape[0]
        srcs = np.arange(n)
        dsts = np.arange(n)
        edge_w = np.ones(n, dtype=np.float32)
    else:
        edge_w = np.abs(W[srcs, dsts]).astype(np.float32)

    x = torch.from_numpy(np.asarray(node_features, dtype=np.float32)).to(dev)
    edge_index = torch.from_numpy(np.stack([srcs, dsts], axis=0)).long().to(dev)
    edge_weight = torch.from_numpy(edge_w).to(dev)
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)


def encode_cdag(
    model: CDAGSage,
    weights: np.ndarray,
    node_features: np.ndarray,
    *,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-inference helper — returns (logits, embeddings) as numpy arrays."""
    dev = device if device is not None else get_device()
    model.eval()
    with torch.no_grad():
        data = cdag_to_pyg_data(weights, node_features, device=dev)
        logits, emb = model(data)
    return logits.detach().cpu().numpy(), emb.detach().cpu().numpy()


def build_model_on_device(cfg: GNNConfig | None = None) -> CDAGSage:
    """Instantiate the GNN, move it to the resolved device, and log VRAM."""
    cfg = cfg or GNNConfig()
    model = CDAGSage(cfg)
    dev = get_device(override=cfg.device if cfg.device != "auto" else None)
    model = model.to(dev)
    log_device_info(dev)
    # Refuse silent CPU fallback if the caller asked for cuda.
    if dev.type == "cuda":
        assert_on_gpu(model, name="CDAGSage")
    return model
