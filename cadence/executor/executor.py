"""Retrain executor with Part-3B EWC mechanics + shadow/canary/rollback.

Consolidates §4.1 (shadow validation), §4.2 (rollback + surrogate feedback),
and the Part-3B partial-fit recipe (target layer freezing + Fisher-weighted
EWC penalty + replay buffer) into one component the RSO can drive without
knowing about the safeguards.

Design:
  * `RetrainExecutor.execute(action, ...)` — one entry point that does:
      - snapshot current adapter state (pre_state) + F1
      - run the requested action (no-op / partial / full)
      - shadow-validate on a held-out recent window
      - promote or rollback based on §4.1's threshold
      - return `ExecutorReport` with (final_state, event_log, was_rolled_back,
        predicted_vs_measured pair for surrogate feedback)
  * Fallbacks live in `cadence.executor.fallbacks` and `.escalation` and are
    composed above the executor (see `RetrainExecutor.execute_with_fallbacks`).

Shadow-mode implementation: instead of "deploy alongside" (which the
production system does), the sandbox scores the freshly-updated model
against a held-out recent window and rolls back if F1 < SLA target. This is
functionally equivalent for our purposes (no live traffic) and matches
Q44-Q45 of the reference.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import f1_score

from cadence.adapters.base import ModelAdapter
from cadence.carbon.model import GridProfile, HardwareProfile, estimate_cost
from cadence.common.logging import get_logger

log = get_logger("cadence.executor.executor")


@dataclass
class ExecutorConfig:
    sla_target: float = 0.65
    # §4.1: shadow window is a held-out recent slice we score the promoted
    # model against before allowing it to serve. Percentage of the retrain-
    # data window is treated as shadow.
    shadow_window_frac: float = 0.25
    # §4.1: promote only if post-fix F1 >= promotion_threshold * sla_target
    promotion_threshold_ratio: float = 1.0
    # §4.2: on rollback, keep the pre_state as the current serving weights
    # (default behaviour).
    keep_pre_state_on_rollback: bool = True
    # Cost model (used by MLflow + the RSO reward).
    finetune_epochs: int = 3
    fullretrain_epochs: int = 10
    replay_buffer_size: int = 5000
    replay_ratio_new: float = 0.7
    ewc_penalty: float = 1000.0
    ewc_fisher_sample_size: int = 2000
    hardware: HardwareProfile = field(default_factory=HardwareProfile)
    grid: GridProfile = field(default_factory=GridProfile)


@dataclass
class ExecutorEvent:
    stage: str  # "action_started", "shadow_scored", "promoted", "rolled_back", "no_op"
    action: str  # "no-op" | "partial" | "full"
    target_layer: str | None
    pre_f1: float
    post_f1: float
    shadow_f1: float
    predicted_post_f1: float | None
    passed_shadow: bool
    was_rolled_back: bool
    gpu_seconds: float
    kg_co2: float
    wall_seconds: float


@dataclass
class ExecutorReport:
    action_taken: str
    event: ExecutorEvent
    surrogate_feedback: tuple[float, float] | None = None
    """(predicted_post_f1, measured_post_f1). §4.2: fed back to surrogate
    training to close the calibration gap over time."""


class RetrainExecutor:
    """Wraps an adapter and drives its retrain lifecycle with §4 safeguards."""

    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        historical_X: np.ndarray,
        historical_y: np.ndarray,
        cfg: ExecutorConfig | None = None,
    ):
        self.adapter = adapter
        self.historical_X = historical_X
        self.historical_y = historical_y
        self.cfg = cfg or ExecutorConfig()
        self._fisher_cache: dict | None = None
        self._theta_star_cache: dict | None = None

    def execute(
        self,
        *,
        action: str,
        window_X: np.ndarray,
        window_y: np.ndarray,
        target_layer: str | None = None,
        predicted_post_f1: float | None = None,
    ) -> ExecutorReport:
        """Run one action end-to-end with shadow + rollback.

        `action`: 'no-op' | 'partial' | 'full'.
        `target_layer`: name of the layer to fine-tune (partial only).
        `predicted_post_f1`: the surrogate's prediction — logged so §4.2
        rollback feedback can compute (predicted, measured) pairs.
        """
        if action not in ("no-op", "partial", "full"):
            raise ValueError(f"unknown action={action!r}")

        # Shadow split: reserve the tail of the window for validation. The
        # retrain sees the head only, so shadow scoring is honestly on data
        # the fine-tune never touched — matches Q44's "held-out recent" split.
        shadow_end = window_X.shape[0]
        shadow_start = int((1 - self.cfg.shadow_window_frac) * shadow_end)
        retrain_X = window_X[:shadow_start]
        retrain_y = window_y[:shadow_start]
        shadow_X = window_X[shadow_start:]
        shadow_y = window_y[shadow_start:]

        pre_state = self.adapter.state_dict()
        pre_threshold = getattr(self.adapter, "decision_threshold", 0.5)
        pre_f1 = _f1_of(self.adapter, window_X, window_y)

        wall_start = time.perf_counter()
        cost_gpu_seconds = 0.0
        cost_kg_co2 = 0.0

        if action == "no-op":
            event = ExecutorEvent(
                stage="no_op",
                action="no-op",
                target_layer=None,
                pre_f1=pre_f1,
                post_f1=pre_f1,
                shadow_f1=pre_f1,
                predicted_post_f1=predicted_post_f1,
                passed_shadow=True,
                was_rolled_back=False,
                gpu_seconds=0.0,
                kg_co2=0.0,
                wall_seconds=0.0,
            )
            log.info(
                "no_op",
                pre_f1=pre_f1,
                sla_target=self.cfg.sla_target,
            )
            return ExecutorReport(
                action_taken="no-op",
                event=event,
                surrogate_feedback=None,
            )

        if action == "partial":
            if target_layer is None:
                raise ValueError("partial action requires target_layer")
            cost = estimate_cost(
                flops_per_forward=self.adapter.flops_per_forward(),
                n_samples=retrain_X.shape[0],
                epochs=self.cfg.finetune_epochs,
                hardware=self.cfg.hardware,
                grid=self.cfg.grid,
            )
            cost_gpu_seconds = cost.gpu_seconds
            cost_kg_co2 = cost.kg_co2
            self._run_partial(retrain_X, retrain_y, target_layer=target_layer)
        else:  # action == "full"
            cost = estimate_cost(
                flops_per_forward=self.adapter.flops_per_forward(),
                n_samples=self.historical_X.shape[0] + retrain_X.shape[0],
                epochs=self.cfg.fullretrain_epochs,
                hardware=self.cfg.hardware,
                grid=self.cfg.grid,
            )
            cost_gpu_seconds = cost.gpu_seconds
            cost_kg_co2 = cost.kg_co2
            self._run_full(retrain_X, retrain_y)

        wall_seconds = time.perf_counter() - wall_start
        post_f1 = _f1_of(self.adapter, window_X, window_y)
        shadow_f1 = _f1_of(self.adapter, shadow_X, shadow_y) if shadow_X.size else post_f1

        # §4.1: shadow gate. Promote only if shadow_f1 >= promotion threshold.
        threshold = self.cfg.promotion_threshold_ratio * self.cfg.sla_target
        passed = shadow_f1 >= threshold
        was_rolled_back = False

        if passed:
            log.info(
                "promoted",
                action=action,
                pre_f1=pre_f1,
                post_f1=post_f1,
                shadow_f1=shadow_f1,
                sla_target=self.cfg.sla_target,
            )
        else:
            was_rolled_back = True
            if self.cfg.keep_pre_state_on_rollback:
                self.adapter.load_state_dict(pre_state)
                if hasattr(self.adapter, "decision_threshold"):
                    self.adapter.decision_threshold = pre_threshold
            log.warning(
                "rolled_back",
                action=action,
                pre_f1=pre_f1,
                post_f1=post_f1,
                shadow_f1=shadow_f1,
                sla_target=self.cfg.sla_target,
            )

        event = ExecutorEvent(
            stage="promoted" if passed else "rolled_back",
            action=action,
            target_layer=target_layer,
            pre_f1=pre_f1,
            post_f1=post_f1,
            shadow_f1=shadow_f1,
            predicted_post_f1=predicted_post_f1,
            passed_shadow=passed,
            was_rolled_back=was_rolled_back,
            gpu_seconds=cost_gpu_seconds,
            kg_co2=cost_kg_co2,
            wall_seconds=wall_seconds,
        )

        # §4.2: rollback feedback for surrogate recalibration.
        feedback = None
        if predicted_post_f1 is not None:
            # The surrogate cares about the F1 on the retrain window (what it
            # predicted), not the shadow slice.
            feedback = (float(predicted_post_f1), float(post_f1))
        return ExecutorReport(
            action_taken=action,
            event=event,
            surrogate_feedback=feedback,
        )

    # ---- private ----

    def _run_partial(
        self,
        retrain_X: np.ndarray,
        retrain_y: np.ndarray,
        *,
        target_layer: str,
    ) -> None:
        # Replay buffer to counter over-fitting to the drifted slice
        # (Part-3B mechanics).
        replay_n = int(self.cfg.replay_buffer_size * (1 - self.cfg.replay_ratio_new))
        replay_idx = np.random.default_rng(0).integers(
            0, self.historical_X.shape[0], size=replay_n
        )
        replay_X = self.historical_X[replay_idx]
        replay_y = self.historical_y[replay_idx]

        fisher_dict = None
        theta_star = None
        if self.cfg.ewc_penalty > 0 and hasattr(self.adapter, "compute_fisher"):
            if self._fisher_cache is not None:
                fisher_dict = self._fisher_cache
                theta_star = self._theta_star_cache
            else:
                sample = min(self.cfg.ewc_fisher_sample_size, self.historical_X.shape[0])
                fisher_dict = self.adapter.compute_fisher(
                    self.historical_X[:sample],
                    self.historical_y[:sample],
                    layers=[target_layer],
                )
                theta_star = {
                    n: p.detach().clone()
                    for n, p in self.adapter._module.named_parameters()
                    if n.startswith(f"{target_layer}.")
                }
                self._fisher_cache = fisher_dict
                self._theta_star_cache = theta_star

        try:
            self.adapter.partial_fit(
                retrain_X,
                retrain_y,
                layers_to_update=[target_layer],
                ewc_penalty=self.cfg.ewc_penalty if fisher_dict is not None else 0.0,
                fisher=fisher_dict,
                theta_star=theta_star,
                replay_X=replay_X,
                replay_y=replay_y,
                max_epochs=self.cfg.finetune_epochs,
            )
        except torch.cuda.OutOfMemoryError as e:
            # §4.7 resource fallback surfaces this exception; the caller
            # (execute_with_fallbacks) catches it and reverts to the pre_state.
            log.error("partial_oom", err=str(e))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def _run_full(self, retrain_X: np.ndarray, retrain_y: np.ndarray) -> None:
        X_full = np.concatenate([self.historical_X, retrain_X], axis=0)
        y_full = np.concatenate([self.historical_y, retrain_y], axis=0)
        try:
            self.adapter.fit(X_full, y_full)
        except torch.cuda.OutOfMemoryError as e:
            log.error("full_oom", err=str(e))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise


def _f1_of(adapter: ModelAdapter, X: np.ndarray, y: np.ndarray) -> float:
    if X.size == 0:
        return 0.0
    probs = adapter.predict_proba(X)
    threshold = getattr(adapter, "decision_threshold", 0.5)
    preds = (probs >= threshold).astype(np.int64)
    return float(f1_score(y, preds, zero_division=0))
