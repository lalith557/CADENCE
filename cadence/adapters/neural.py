"""FraudNet — the neural adapter, exactly per D-1 and Part 3 Table A of the reference.

Architecture (input_dim → 64 ReLU → 32 ReLU → 1 Sigmoid) is a decision (D-1); the
adapter itself is model-family-generic, so you can also instantiate wider/deeper
MLPs by passing a different `hidden_dims`.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from cadence.adapters.base import FitResult, ModelAdapter, TrainMetrics
from cadence.common.logging import get_logger

log = get_logger("cadence.adapters.neural")


@dataclass
class FraudNetConfig:
    input_dim: int
    hidden_dims: list[int]
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 30
    early_stopping_patience: int = 5
    class_weighted: bool = True
    device: str = "auto"  # "auto" -> cuda if available else cpu


class _FraudNetModule(nn.Module):
    """Pure nn.Module — the adapter wraps this."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        prev = input_dim
        self._layer_names: list[str] = []
        for i, h in enumerate(hidden_dims):
            name = f"layer{i + 1}"
            linear = nn.Linear(prev, h)
            # Register with the exact name the adapter surfaces, so
            # state_dict keys map cleanly to layer names.
            self.add_module(name, linear)
            self.add_module(f"{name}_relu", nn.ReLU())
            self.add_module(f"{name}_dropout", nn.Dropout(dropout))
            self._layer_names.append(name)
            prev = h
        self.add_module("output", nn.Linear(prev, 1))
        self._layer_names.append("output")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns (logits, {layer_name: post-relu activations}) in one pass.

        The dict is what the CDAG builder reads. Output layer is intentionally
        NOT in the trace — it collapses to a single scalar and isn't a useful
        activation-cluster node.
        """
        traces: dict[str, torch.Tensor] = {}
        h = x
        for name in self._layer_names[:-1]:
            linear = getattr(self, name)
            relu = getattr(self, f"{name}_relu")
            drop = getattr(self, f"{name}_dropout")
            h = drop(relu(linear(h)))
            # Store the post-ReLU vector for CDAG's k-means (D-2).
            traces[name] = h.detach()
        logits = self.output(h)
        return logits.squeeze(-1), traces


class FraudNet(ModelAdapter):
    family = "neural_mlp"

    def __init__(self, cfg: FraudNetConfig):
        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)
        self._module = _FraudNetModule(cfg.input_dim, cfg.hidden_dims, cfg.dropout).to(self.device)
        # Exposed to CDAG/tap machinery.
        self.tap_layer_names = list(self._module._layer_names[:-1])  # exclude output
        self._optimizer: torch.optim.Optimizer | None = None
        # Decision threshold — tuned on validation after fit(). For extreme
        # class imbalance (Fraud is 0.17% positive), 0.5 is a bad default.
        self.decision_threshold: float = 0.5

    @staticmethod
    def _resolve_device(spec: str) -> torch.device:
        if spec == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(spec)

    # ---- ModelAdapter API ----

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        **kwargs: Any,
    ) -> FitResult:
        pos_weight = None
        if self.cfg.class_weighted:
            n_pos = float((y == 1).sum())
            n_neg = float((y == 0).sum())
            if n_pos > 0:
                pos_weight = torch.tensor([n_neg / n_pos], device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self._optimizer = torch.optim.Adam(self._module.parameters(), lr=self.cfg.lr)

        train_loader = _make_loader(X, y, self.cfg.batch_size, shuffle=True)
        val_pack = (X_val, y_val) if X_val is not None and y_val is not None else None

        best_val_f1 = -1.0
        best_epoch = 0
        best_state: dict[str, Any] | None = None
        patience_ctr = 0
        history: list[TrainMetrics] = []

        wall_start = time.perf_counter()
        gpu_time = 0.0

        for epoch in range(1, self.cfg.max_epochs + 1):
            self._module.train()
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True).float()
                self._optimizer.zero_grad(set_to_none=True)
                logits, _ = self._module(xb)
                loss = criterion(logits, yb)
                loss.backward()
                self._optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            epoch_loss /= max(n_batches, 1)
            epoch_wall = time.perf_counter() - epoch_start
            if self.device.type == "cuda":
                gpu_time += epoch_wall

            train_f1 = self._score_f1(X, y)
            if val_pack is not None:
                val_f1 = self._score_f1(*val_pack)
                val_auroc = self._score_auroc(*val_pack)
            else:
                val_f1 = train_f1
                val_auroc = float("nan")

            m = TrainMetrics(
                loss=epoch_loss,
                train_f1=train_f1,
                val_f1=val_f1,
                val_auroc=val_auroc,
                epoch=epoch,
                epoch_time_s=epoch_wall,
            )
            history.append(m)
            log.info("epoch", **m.__dict__)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_state = self.state_dict()
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.cfg.early_stopping_patience:
                    log.info(
                        "early_stop", epoch=epoch, best_epoch=best_epoch, best_val_f1=best_val_f1
                    )
                    break

        if best_state is not None:
            self.load_state_dict(best_state)

        # Tune the decision threshold on validation (or train if no val given).
        # For extreme imbalance, F1 at threshold 0.5 is systematically bad.
        tune_X, tune_y = (X_val, y_val) if val_pack is not None else (X, y)
        self.decision_threshold = self._tune_threshold(tune_X, tune_y)
        # Recompute best_val_f1 with the tuned threshold.
        best_val_f1 = self._score_f1(tune_X, tune_y)

        return FitResult(
            epochs_run=history[-1].epoch,
            final_metrics=history[-1],
            best_val_f1=best_val_f1,
            best_epoch=best_epoch,
            total_wall_s=time.perf_counter() - wall_start,
            total_gpu_time_s=gpu_time,
        )

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        layers_to_update: list[str] | None = None,
        ewc_penalty: float = 0.0,
        replay_X: np.ndarray | None = None,
        replay_y: np.ndarray | None = None,
        fisher: dict[str, torch.Tensor] | None = None,
        theta_star: dict[str, torch.Tensor] | None = None,
        max_epochs: int = 10,
        finetune_lr: float = 1e-4,
        **kwargs: Any,
    ) -> FitResult:
        """Partial retrain per Part 3B: freeze non-target layers + EWC penalty."""
        # Assemble the fine-tuning dataset.
        if replay_X is not None and replay_y is not None:
            X_ft = np.concatenate([X, replay_X], axis=0)
            y_ft = np.concatenate([y, replay_y], axis=0)
        else:
            X_ft, y_ft = X, y

        # Freeze non-target layers.
        if layers_to_update is not None:
            for name, module in self._module.named_children():
                stem = name.split("_")[0]
                trainable = stem in layers_to_update
                for p in module.parameters():
                    p.requires_grad = trainable
        else:
            for p in self._module.parameters():
                p.requires_grad = True

        trainable_params = [p for p in self._module.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("partial_fit: no trainable parameters after freezing")

        n_pos = float((y_ft == 1).sum())
        n_neg = float((y_ft == 0).sum())
        pos_weight = None
        if self.cfg.class_weighted and n_pos > 0:
            pos_weight = torch.tensor([n_neg / n_pos], device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self._optimizer = torch.optim.Adam(trainable_params, lr=finetune_lr)

        loader = _make_loader(X_ft, y_ft, self.cfg.batch_size, shuffle=True)

        best_val_f1 = -1.0
        best_epoch = 0
        best_state = self.state_dict()
        history: list[TrainMetrics] = []
        wall_start = time.perf_counter()
        gpu_time = 0.0

        for epoch in range(1, max_epochs + 1):
            self._module.train()
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True).float()
                self._optimizer.zero_grad(set_to_none=True)
                logits, _ = self._module(xb)
                loss = criterion(logits, yb)

                if ewc_penalty > 0 and fisher is not None and theta_star is not None:
                    ewc = torch.zeros((), device=self.device)
                    for name, p in self._module.named_parameters():
                        if name in fisher and name in theta_star:
                            ewc = ewc + (fisher[name] * (p - theta_star[name]).pow(2)).sum()
                    loss = loss + ewc_penalty * ewc

                loss.backward()
                self._optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            epoch_loss /= max(n_batches, 1)
            epoch_wall = time.perf_counter() - epoch_start
            if self.device.type == "cuda":
                gpu_time += epoch_wall

            train_f1 = self._score_f1(X, y)
            m = TrainMetrics(
                loss=epoch_loss,
                train_f1=train_f1,
                val_f1=train_f1,
                val_auroc=float("nan"),
                epoch=epoch,
                epoch_time_s=epoch_wall,
            )
            history.append(m)
            if train_f1 > best_val_f1:
                best_val_f1 = train_f1
                best_epoch = epoch
                best_state = self.state_dict()

        self.load_state_dict(best_state)

        # Re-tune the decision threshold on the new post-fine-tune distribution.
        self.decision_threshold = self._tune_threshold(X, y)

        # Restore requires_grad=True on everything for future full retrains.
        for p in self._module.parameters():
            p.requires_grad = True

        return FitResult(
            epochs_run=history[-1].epoch if history else 0,
            final_metrics=history[-1] if history else TrainMetrics(0, 0, 0, 0, 0, 0),
            best_val_f1=best_val_f1,
            best_epoch=best_epoch,
            total_wall_s=time.perf_counter() - wall_start,
            total_gpu_time_s=gpu_time,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._module.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X.astype(np.float32)).to(self.device)
            logits, _ = self._module(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs.astype(np.float32)

    def forward_with_traces(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        self._module.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X.astype(np.float32)).to(self.device)
            logits, traces = self._module(xb)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            traces_np = {k: v.cpu().numpy().astype(np.float32) for k, v in traces.items()}
        return probs, traces_np

    def clone(self) -> FraudNet:
        new = FraudNet(self.cfg)
        new.load_state_dict(copy.deepcopy(self.state_dict()))
        return new

    def state_dict(self) -> dict[str, Any]:
        return {k: v.detach().clone() for k, v in self._module.state_dict().items()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._module.load_state_dict(state)

    def get_layer_params(self, layer_name: str) -> dict[str, torch.Tensor]:
        return {
            name: p
            for name, p in self._module.named_parameters()
            if name.startswith(f"{layer_name}.")
        }

    def n_trainable_params(self, layers: list[str] | None = None) -> int:
        if layers is None:
            return sum(p.numel() for p in self._module.parameters())
        total = 0
        for name, p in self._module.named_parameters():
            stem = name.split(".")[0]
            if stem in layers:
                total += p.numel()
        return total

    def flops_per_forward(self) -> int:
        """2 * sum(in*out) for each linear layer — the standard MAC-count convention."""
        total = 0
        prev = self.cfg.input_dim
        for h in self.cfg.hidden_dims:
            total += 2 * prev * h
            prev = h
        total += 2 * prev * 1
        return total

    # ---- helpers ----

    def _score_f1(self, X: np.ndarray, y: np.ndarray) -> float:
        probs = self.predict_proba(X)
        preds = (probs >= self.decision_threshold).astype(np.int64)
        return float(f1_score(y, preds, zero_division=0))

    def _tune_threshold(self, X: np.ndarray, y: np.ndarray, *, n_grid: int = 101) -> float:
        """Pick threshold maximizing F1 on (X, y). For imbalanced binary tasks,
        the 0.5 default is a terrible choice — see W-16 in weaknesses.md."""
        probs = self.predict_proba(X)
        if (y == 1).sum() == 0:
            return 0.5
        # Search over percentiles of probs to skip regions of no data.
        candidates = np.quantile(probs, np.linspace(0.5, 0.999, n_grid))
        candidates = np.unique(np.clip(candidates, 1e-4, 1 - 1e-4))
        best_f1 = -1.0
        best_t = 0.5
        for t in candidates:
            preds = (probs >= t).astype(np.int64)
            f1 = f1_score(y, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = float(f1)
                best_t = float(t)
        return best_t

    def _score_auroc(self, X: np.ndarray, y: np.ndarray) -> float:
        try:
            probs = self.predict_proba(X)
            return float(roc_auc_score(y, probs))
        except ValueError:
            return float("nan")

    def compute_fisher(
        self, X: np.ndarray, y: np.ndarray, layers: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        """Diagonal Fisher information over a data sample. Cached per model per D-8."""
        self._module.train()
        criterion = nn.BCEWithLogitsLoss()
        fisher = {name: torch.zeros_like(p) for name, p in self._module.named_parameters()}

        loader = _make_loader(X, y, self.cfg.batch_size, shuffle=False)
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True).float()
            self._module.zero_grad(set_to_none=True)
            logits, _ = self._module(xb)
            loss = criterion(logits, yb)
            loss.backward()
            for name, p in self._module.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad.detach().pow(2) * xb.shape[0]
            n_seen += xb.shape[0]

        for name in fisher:
            fisher[name] /= max(n_seen, 1)

        if layers is not None:
            fisher = {n: f for n, f in fisher.items() if n.split(".")[0] in layers}
        return fisher


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, *, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
