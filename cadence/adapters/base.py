"""ModelAdapter — the interface that lets CADENCE watch heterogeneous production models.

The single most important product-quality piece of CADENCE: everything below
here (drift trigger, CDAG builder, GNN, surrogate, RSO, executor) targets this
interface, never a concrete framework. Adding a new model family (LightGBM,
sklearn LR, custom torch module) means writing one adapter, nothing else.

Key operations:
- fit / partial_fit  — supervised update against (X, y).
- predict_proba      — for scoring + F1 measurement.
- forward_with_traces — return the tuple (predictions, activations-per-layer)
  in a single pass; used by the collector so we never re-forward.
- get_layer_params / freeze_all_except — surface the parameter-name → tensor
  map, so the executor can freeze non-target layers per D-8.
- clone              — deep-copy for sandbox experiments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TrainMetrics:
    """What every trainer produces per epoch."""

    loss: float
    train_f1: float
    val_f1: float
    val_auroc: float
    epoch: int
    epoch_time_s: float


@dataclass
class FitResult:
    """Summary of a full training run."""

    epochs_run: int
    final_metrics: TrainMetrics
    best_val_f1: float
    best_epoch: int
    total_wall_s: float
    total_gpu_time_s: float


class ModelAdapter(ABC):
    """Anything CADENCE watches implements this."""

    #: human-readable adapter name, used in MLflow tags
    family: str = "abstract"
    #: layer names *in the order signals flow* — used by the collector + CDAG
    #: to know where "Layer 1" / "Layer 2" (in the walkthrough's sense) live.
    tap_layer_names: list[str] = []

    @abstractmethod
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
        """Full training from scratch."""

    @abstractmethod
    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        layers_to_update: list[str] | None = None,
        ewc_penalty: float = 0.0,
        replay_X: np.ndarray | None = None,
        replay_y: np.ndarray | None = None,
        **kwargs: Any,
    ) -> FitResult:
        """Continual / partial update. `layers_to_update=None` means all layers.

        Adapters that don't support layer-level partial updates (e.g. a monolithic
        tree ensemble) must raise `NotImplementedError` — the executor catches
        that and falls back to full retrain, logging the reason.
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(y=1|x) as float32 [N] for binary, or [N, K] for multiclass."""

    @abstractmethod
    def forward_with_traces(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return (probabilities, {layer_name: activation_matrix}).

        Activation matrix shape is (n_rows, layer_width). Layer names must be
        drawn from `tap_layer_names`. Adapters that have no meaningful notion
        of internal activations (LR, XGBoost) should return an empty dict —
        the CDAG builder degrades to a features-only graph in that case.
        """

    @abstractmethod
    def clone(self) -> ModelAdapter:
        """Deep copy — used for sandbox counterfactual experiments."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Framework-native state dict; the executor snapshots this before a partial retrain."""

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None: ...

    @abstractmethod
    def get_layer_params(self, layer_name: str) -> dict[str, Any]:
        """Return {parameter_name: tensor-like} for the named layer."""

    @abstractmethod
    def n_trainable_params(self, layers: list[str] | None = None) -> int:
        """Total trainable parameter count, optionally scoped to specific layers.

        Used by the D-12 "diffuse responsibility → escalate to full" rule.
        """

    @abstractmethod
    def flops_per_forward(self) -> int:
        """Approximate FLOPs for a single forward pass.

        Used by the RSO's compute-cost model. Approximation is fine — the RL
        agent only needs *relative* costs across actions, not absolute FLOPs.
        """
