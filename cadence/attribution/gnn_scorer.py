"""GNN-based responsibility scorer + pretraining loop.

Drop-in replacement for `PSIResponsibilityScorer` and the pre-Step-B
`CDAGResponsibilityScorer` (structural proxy). Implements the `ResponsibilityScorer`
protocol so RSO does not need retraining when we swap this in.

Pipeline at score time:
  1. Build CDAG for the current window (same as the structural scorer).
  2. Encode the CDAG + per-node stats through the trained GNN.
  3. Take the readout logits at the FEATURE NODES ONLY.
  4. Softmax over feature nodes → per-feature responsibility in [0, 1] summing to 1.
  5. Rescale so max = 1, matching the [0, 1] convention of the other scorers.

Pipeline at pretrain time:
  1. `generate_sandbox_dataset(...)` produces (CDAG, ground-truth feature)
     tuples for many synthetic drift injections.
  2. `train_gnn(...)` trains the GNN via cross-entropy over feature-node logits,
     with the ground-truth feature-node index as the target class.
  3. All tensors sit on GPU; per-epoch peak VRAM logged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from cadence.adapters.base import ModelAdapter
from cadence.attribution.gnn import CDAGSage, GNNConfig, build_model_on_device, cdag_to_pyg_data
from cadence.attribution.sandbox_labels import SandboxSample
from cadence.cdag import (
    ActivationTap,
    CDAGNodeSet,
    build_cdag_from_windows,
    compute_windowed_signals,
)
from cadence.common.device import (
    assert_on_gpu,
    cuda_memory_snapshot,
    get_device,
    log_device_info,
    reset_peak_memory,
)
from cadence.common.logging import get_logger

log = get_logger("cadence.attribution.gnn_scorer")


@dataclass
class GNNTrainConfig:
    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 32
    val_frac: float = 0.2
    seed: int = 42


def train_gnn(
    model: CDAGSage,
    samples: list[SandboxSample],
    cfg: GNNTrainConfig | None = None,
    *,
    device: torch.device | None = None,
) -> dict:
    """Supervised pretrain of the GNN on the sandbox dataset.

    Objective: cross-entropy on feature-node logits, target = root-cause
    feature-node index. This teaches the readout to concentrate mass on the
    node whose input distribution changed. Even without the surrogate (Step
    C), this signal produces attribution scores that beat correlational PSI.
    """
    cfg = cfg or GNNTrainConfig()
    dev = device if device is not None else get_device()

    if dev.type == "cuda":
        reset_peak_memory()
        log_device_info(dev)
        assert_on_gpu(model, name="CDAGSage[train]")

    rng = np.random.default_rng(cfg.seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_val = max(1, int(cfg.val_frac * len(idx)))
    val_idx = set(idx[:n_val].tolist())
    train_idx = [int(i) for i in idx[n_val:]]
    val_idx_list = [int(i) for i in idx[:n_val]]

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: list[dict] = []

    def _epoch(loader_idx: list[int], train: bool) -> tuple[float, float]:
        model.train(mode=train)
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        # Simple manual mini-batching: one graph per optimizer step to keep
        # things predictable on tiny graphs. PyG's Batch could bundle multiple
        # graphs, but each graph is O(40 nodes) so per-step overhead is fine.
        for j, i in enumerate(loader_idx):
            s = samples[i]
            data: Data = s.data
            if train:
                opt.zero_grad(set_to_none=True)
            logits, _ = model(data)
            # Restrict cross-entropy to feature nodes only.
            n_feat = _n_feature_nodes(samples)
            feat_logits = logits[:n_feat].unsqueeze(0)  # (1, n_feat)
            target = torch.tensor([s.root_cause_feature_idx], device=dev, dtype=torch.long)
            loss = F.cross_entropy(feat_logits, target)
            if train:
                loss.backward()
                opt.step()
            total_loss += float(loss.item())
            pred = int(feat_logits.argmax(dim=-1).item())
            total_correct += int(pred == s.root_cause_feature_idx)
            total_n += 1
        return total_loss / max(total_n, 1), total_correct / max(total_n, 1)

    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = _epoch(train_idx, train=True)
        val_loss, val_acc = _epoch(val_idx_list, train=False)
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(entry)
        log.info("gnn_epoch", **entry)

    summary = {
        "wall_s": time.perf_counter() - t0,
        "final_val_acc": history[-1]["val_acc"],
        "final_val_loss": history[-1]["val_loss"],
        "history": history,
        "n_train": len(train_idx),
        "n_val": len(val_idx_list),
    }
    if dev.type == "cuda":
        snap = cuda_memory_snapshot()
        summary["max_vram_mb"] = round(snap["max_allocated"] / 1024**2, 3)
        summary["allocated_vram_mb"] = round(snap["allocated"] / 1024**2, 3)
        log.info(
            "gnn_train_done",
            wall_s=summary["wall_s"],
            final_val_acc=summary["final_val_acc"],
            max_vram_mb=summary["max_vram_mb"],
        )
    else:
        log.info("gnn_train_done", wall_s=summary["wall_s"], final_val_acc=summary["final_val_acc"])
    return summary


def _n_feature_nodes(samples: list[SandboxSample]) -> int:
    """Recover the number of feature nodes from the labelled samples.

    Every sample's `root_cause_feature_idx` is 0..(n_features-1), so the max
    over a well-covered training set + 1 is a safe estimate. If the caller
    knows n_features exactly they can also monkey-patch this — but with a
    reasonable sandbox size it's always right.
    """
    if not samples:
        return 0
    return max(s.root_cause_feature_idx for s in samples) + 1


@dataclass
class GNNResponsibilityScorer:
    """Learned attribution scorer — score(f) ∝ softmax over trained GNN readouts."""

    model: CDAGSage
    tap: ActivationTap
    node_set: CDAGNodeSet
    baseline_means: np.ndarray
    baseline_stds: np.ndarray
    window_size: int = 512
    pc_alpha: float = 0.05
    notears_lambda: float = 0.05
    notears_max_iter: int = 25
    device: torch.device | None = None
    name: str = "gnn_learned"

    @classmethod
    def build(
        cls,
        *,
        adapter: ModelAdapter,
        baseline_X: np.ndarray,
        feature_names: list[str],
        baseline_f1: float,
        k_overrides: dict[str, int] | None = None,
        window_size: int = 512,
        gnn_cfg: GNNConfig | None = None,
    ) -> "GNNResponsibilityScorer":
        """Build node-set + tap on baseline data; instantiate an untrained GNN.
        The caller invokes `train_gnn(...)` next.
        """
        from cadence.cdag.nodes import build_baseline_stats, build_node_set

        tap = ActivationTap(adapter).fit(baseline_X, k_overrides=k_overrides)
        node_set = build_node_set(feature_names, tap)
        bmean, bstd = build_baseline_stats(baseline_X, tap, node_set, baseline_f1)
        model = build_model_on_device(gnn_cfg or GNNConfig())
        return cls(
            model=model,
            tap=tap,
            node_set=node_set,
            baseline_means=bmean,
            baseline_stds=bstd,
            window_size=window_size,
            device=get_device(),
        )

    def __call__(
        self, adapter: ModelAdapter, window_X: np.ndarray, window_y: np.ndarray
    ) -> np.ndarray:
        from sklearn.metrics import f1_score

        threshold = getattr(adapter, "decision_threshold", 0.5)
        probs = adapter.predict_proba(window_X)
        preds = (probs >= threshold).astype(np.int64)
        current_f1 = float(f1_score(window_y, preds, zero_division=0))

        # Reuse existing tap topology on the (possibly-updated) adapter.
        self.tap.adapter = adapter

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
        node_features = signals[-1]  # (n_nodes, 4)
        data = cdag_to_pyg_data(cdag.weights, node_features, device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(data)
        n_feat = len(self.node_set.feature_indices)
        feat_logits = logits[:n_feat].detach().cpu().numpy().astype(np.float32)

        # Softmax over feature nodes → probability mass; rescale to [0, 1].
        z = feat_logits - feat_logits.max()
        p = np.exp(z)
        p = p / (p.sum() + 1e-9)
        p = p / (p.max() + 1e-9)
        return p.astype(np.float32)
