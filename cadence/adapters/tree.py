"""LightGBM tree-ensemble adapter — Step F generality across model families.

Satisfies the `ModelAdapter` protocol so every CADENCE piece downstream
(PSI trigger, `RetrainExecutor`, `EscalationLadder`, PSI-scorer path)
treats a LightGBM model exactly like FraudNet.

Design notes:
  * `partial_fit(layers_to_update=[...])` raises NotImplementedError — trees
    don't have layer-level partial updates. The executor catches this and
    falls back to full retrain, per `ModelAdapter.partial_fit`'s contract.
  * `partial_fit(layers_to_update=None)` DOES work: it warm-starts a LightGBM
    booster with `init_model=self.booster` and trains a small number of
    additional rounds. This is a genuine partial retrain for trees.
  * `forward_with_traces` returns an empty traces dict — trees have no
    activations. The CDAG builder degrades to a feature-only graph on this
    adapter (per Q104 of the reference / §4.8 cold-start fallback).
  * `flops_per_forward` uses num_leaves * num_trees * 32 as a rough cost —
    the RSO's reward only needs relative cost, not absolute FLOPs.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from cadence.adapters.base import FitResult, ModelAdapter, TrainMetrics
from cadence.common.logging import get_logger

log = get_logger("cadence.adapters.tree")


@dataclass
class LightGBMConfig:
    num_leaves: int = 31
    max_depth: int = -1
    learning_rate: float = 0.05
    n_estimators: int = 200
    min_child_samples: int = 20
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    class_weight: str | None = "balanced"
    partial_boost_rounds: int = 50  # for partial_fit warm-start
    seed: int = 42


class LightGBMAdapter(ModelAdapter):
    family = "tree_lightgbm"

    def __init__(self, cfg: LightGBMConfig | None = None):
        self.cfg = cfg or LightGBMConfig()
        self.booster: lgb.Booster | None = None
        self._input_dim: int | None = None
        self.decision_threshold: float = 0.5
        # No tap layers for trees — the CDAG falls back to feature-only.
        self.tap_layer_names = []

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
        self._input_dim = int(X.shape[1])
        params = self._lgb_params()
        train_set = lgb.Dataset(X, label=y, weight=sample_weight)
        valid_sets: list[lgb.Dataset] = []
        if X_val is not None and y_val is not None:
            valid_sets.append(lgb.Dataset(X_val, label=y_val, reference=train_set))

        t0 = time.perf_counter()
        self.booster = lgb.train(
            params,
            train_set,
            num_boost_round=self.cfg.n_estimators,
            valid_sets=valid_sets if valid_sets else None,
            callbacks=[lgb.log_evaluation(period=0)],
        )
        wall = time.perf_counter() - t0

        # Threshold tuning matches FraudNet's F1 sweep so imbalanced datasets
        # (Fraud, GMSC) don't get pinned to 0.5.
        tune_X = X_val if X_val is not None else X
        tune_y = y_val if y_val is not None else y
        self.decision_threshold = self._tune_threshold(tune_X, tune_y)

        val_f1 = self._score_f1(tune_X, tune_y)
        val_auroc = self._score_auroc(tune_X, tune_y)
        metrics = TrainMetrics(
            loss=0.0,
            train_f1=self._score_f1(X, y),
            val_f1=val_f1,
            val_auroc=val_auroc,
            epoch=self.cfg.n_estimators,
            epoch_time_s=wall,
        )
        log.info(
            "lgb_fit_done",
            n_estimators=self.cfg.n_estimators,
            val_f1=val_f1,
            val_auroc=val_auroc,
            tuned_threshold=self.decision_threshold,
            wall_s=wall,
        )
        return FitResult(
            epochs_run=self.cfg.n_estimators,
            final_metrics=metrics,
            best_val_f1=val_f1,
            best_epoch=self.cfg.n_estimators,
            total_wall_s=wall,
            total_gpu_time_s=0.0,  # LightGBM CPU by default
        )

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        layers_to_update: list[str] | None = None,
        ewc_penalty: float = 0.0,  # noqa: ARG002 — trees have no Fisher analogue
        replay_X: np.ndarray | None = None,
        replay_y: np.ndarray | None = None,
        fisher: dict | None = None,  # noqa: ARG002
        theta_star: dict | None = None,  # noqa: ARG002
        max_epochs: int | None = None,
        **kwargs: Any,
    ) -> FitResult:
        """Warm-start LightGBM boosting from the current booster.

        Trees have no per-layer partial update, so a non-None `layers_to_update`
        raises NotImplementedError — matching the ModelAdapter contract. The
        executor catches that and falls back to a full retrain, logging the
        `adapter_no_partial_fit_fallback_full` reason.
        """
        if layers_to_update is not None:
            raise NotImplementedError(
                f"{self.family}: no layer-level partial retrain — call fit() instead"
            )
        if self.booster is None:
            raise RuntimeError("LightGBMAdapter.partial_fit called before fit()")

        # Concatenate replay buffer + new data (Part 3B / D-8 replay ratio).
        if replay_X is not None and replay_y is not None:
            X_ft = np.concatenate([X, replay_X], axis=0)
            y_ft = np.concatenate([y, replay_y], axis=0)
        else:
            X_ft, y_ft = X, y

        boost_rounds = max_epochs if max_epochs is not None else self.cfg.partial_boost_rounds
        params = self._lgb_params()
        train_set = lgb.Dataset(X_ft, label=y_ft)

        t0 = time.perf_counter()
        self.booster = lgb.train(
            params,
            train_set,
            num_boost_round=boost_rounds,
            init_model=self.booster,
            callbacks=[lgb.log_evaluation(period=0)],
        )
        wall = time.perf_counter() - t0

        self.decision_threshold = self._tune_threshold(X, y)
        val_f1 = self._score_f1(X, y)
        metrics = TrainMetrics(
            loss=0.0,
            train_f1=val_f1,
            val_f1=val_f1,
            val_auroc=self._score_auroc(X, y),
            epoch=boost_rounds,
            epoch_time_s=wall,
        )
        return FitResult(
            epochs_run=boost_rounds,
            final_metrics=metrics,
            best_val_f1=val_f1,
            best_epoch=boost_rounds,
            total_wall_s=wall,
            total_gpu_time_s=0.0,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("LightGBMAdapter.predict_proba called before fit()")
        probs = self.booster.predict(X)
        return np.asarray(probs, dtype=np.float32)

    def forward_with_traces(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        # Trees have no meaningful "activation" — return probs + empty dict.
        return self.predict_proba(X), {}

    def clone(self) -> LightGBMAdapter:
        new = LightGBMAdapter(copy.deepcopy(self.cfg))
        new._input_dim = self._input_dim
        new.decision_threshold = self.decision_threshold
        if self.booster is not None:
            new.booster = self.booster.__class__(model_str=self.booster.model_to_string())
        return new

    def state_dict(self) -> dict[str, Any]:
        return {
            "model_str": self.booster.model_to_string() if self.booster is not None else None,
            "decision_threshold": float(self.decision_threshold),
            "input_dim": self._input_dim,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        mstr = state.get("model_str")
        if mstr is None:
            self.booster = None
        else:
            self.booster = lgb.Booster(model_str=mstr)
        self.decision_threshold = float(state.get("decision_threshold", 0.5))
        self._input_dim = state.get("input_dim")

    def get_layer_params(self, layer_name: str) -> dict[str, Any]:
        # Trees expose no layer-level parameters; return empty so callers can
        # skip layer-targeted operations gracefully.
        return {}

    def n_trainable_params(self, layers: list[str] | None = None) -> int:
        if self.booster is None:
            return 0
        # Approximate: num_trees × num_leaves × 2 (split threshold + leaf value).
        return int(self.booster.num_trees()) * int(self.cfg.num_leaves) * 2

    def flops_per_forward(self) -> int:
        if self.booster is None:
            return 0
        # Each tree evaluates ~log2(num_leaves) comparisons; conservative bound.
        n_trees = int(self.booster.num_trees())
        depth = max(1, int(np.log2(max(self.cfg.num_leaves, 2))))
        return n_trees * depth * 4  # 4 ops per node

    # ---- helpers ----

    def _lgb_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "num_leaves": self.cfg.num_leaves,
            "max_depth": self.cfg.max_depth,
            "learning_rate": self.cfg.learning_rate,
            "min_data_in_leaf": self.cfg.min_child_samples,
            "lambda_l1": self.cfg.reg_alpha,
            "lambda_l2": self.cfg.reg_lambda,
            "bagging_fraction": self.cfg.subsample,
            "bagging_freq": 5,
            "feature_fraction": self.cfg.colsample_bytree,
            "verbose": -1,
            "seed": self.cfg.seed,
        }
        if self.cfg.class_weight == "balanced":
            params["is_unbalance"] = True
        return params

    def _score_f1(self, X: np.ndarray, y: np.ndarray) -> float:
        probs = self.predict_proba(X)
        preds = (probs >= self.decision_threshold).astype(np.int64)
        return float(f1_score(y, preds, zero_division=0))

    def _score_auroc(self, X: np.ndarray, y: np.ndarray) -> float:
        try:
            return float(roc_auc_score(y, self.predict_proba(X)))
        except ValueError:
            return float("nan")

    def _tune_threshold(self, X: np.ndarray, y: np.ndarray, *, n_grid: int = 101) -> float:
        probs = self.predict_proba(X)
        if (y == 1).sum() == 0:
            return 0.5
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
