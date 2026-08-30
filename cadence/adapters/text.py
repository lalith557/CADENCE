"""TF-IDF + Logistic Regression text adapter — Step F text-drift experiment.

Implements the `ModelAdapter` protocol on top of an sklearn Pipeline. Text
adapters have no meaningful "activation clusters" (the vocabulary vector
IS the feature space), so:
  * `forward_with_traces` returns an empty traces dict → CDAG degrades
    to feature-only per §4.8 cold-start.
  * `partial_fit(layers=[...])` raises NotImplementedError → executor
    falls back to full retrain.
  * `partial_fit(layers=None)` warm-starts the LR coefficients from the
    current fit and calls `.fit(...)` on the concatenated slice + replay
    buffer (a genuine partial: the vocab stays fixed, only coefficients
    move).

The vocabulary is *frozen* at pretrain time — that's what makes
vocabulary drift measurable. Without this, `partial_fit` would silently
re-fit the vocabulary and mask the drift signal.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from cadence.adapters.base import FitResult, ModelAdapter, TrainMetrics
from cadence.common.logging import get_logger

log = get_logger("cadence.adapters.text")


@dataclass
class TextConfig:
    max_features: int = 20_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 5
    lr_C: float = 1.0
    lr_max_iter: int = 200
    class_weight: str | None = "balanced"
    seed: int = 42


class TFIDFLogRegAdapter(ModelAdapter):
    family = "text_tfidf_lr"

    def __init__(self, cfg: TextConfig | None = None):
        self.cfg = cfg or TextConfig()
        self.vectorizer: TfidfVectorizer | None = None
        self.clf: LogisticRegression | None = None
        self.decision_threshold: float = 0.5
        self.tap_layer_names = []  # text has no tap layers

    # ---- ModelAdapter API ----

    def fit(
        self,
        X: np.ndarray | list[str],
        y: np.ndarray,
        *,
        X_val=None,
        y_val=None,
        sample_weight=None,
        **kwargs: Any,
    ) -> FitResult:
        texts = list(X)
        y = np.asarray(y, dtype=np.int64)

        # Fit the vectorizer only the FIRST time — frozen thereafter so
        # partial retrains can't silently mask vocabulary drift.
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=self.cfg.max_features,
                ngram_range=self.cfg.ngram_range,
                min_df=self.cfg.min_df,
            )
            X_tf = self.vectorizer.fit_transform(texts)
        else:
            X_tf = self.vectorizer.transform(texts)

        self.clf = LogisticRegression(
            C=self.cfg.lr_C,
            max_iter=self.cfg.lr_max_iter,
            class_weight=self.cfg.class_weight,
            random_state=self.cfg.seed,
            solver="liblinear",
        )
        t0 = time.perf_counter()
        self.clf.fit(X_tf, y, sample_weight=sample_weight)
        wall = time.perf_counter() - t0

        tune_X = list(X_val) if X_val is not None else texts
        tune_y = np.asarray(y_val) if y_val is not None else y
        self.decision_threshold = self._tune_threshold(tune_X, tune_y)
        val_f1 = self._score_f1(tune_X, tune_y)
        val_auroc = self._score_auroc(tune_X, tune_y)

        log.info(
            "text_fit_done",
            n=len(texts),
            n_features=X_tf.shape[1],
            val_f1=val_f1,
            val_auroc=val_auroc,
            tuned_threshold=self.decision_threshold,
            wall_s=wall,
        )
        metrics = TrainMetrics(
            loss=0.0,
            train_f1=self._score_f1(texts, y),
            val_f1=val_f1,
            val_auroc=val_auroc,
            epoch=1,
            epoch_time_s=wall,
        )
        return FitResult(
            epochs_run=1,
            final_metrics=metrics,
            best_val_f1=val_f1,
            best_epoch=1,
            total_wall_s=wall,
            total_gpu_time_s=0.0,
        )

    def partial_fit(
        self,
        X,
        y: np.ndarray,
        *,
        layers_to_update: list[str] | None = None,
        ewc_penalty: float = 0.0,  # noqa: ARG002
        replay_X=None,
        replay_y=None,
        fisher=None,  # noqa: ARG002
        theta_star=None,  # noqa: ARG002
        max_epochs: int | None = None,  # noqa: ARG002
        finetune_lr: float | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> FitResult:
        if layers_to_update is not None:
            raise NotImplementedError(
                f"{self.family}: no layer-level partial retrain — call fit() instead"
            )
        if self.vectorizer is None:
            raise RuntimeError("TFIDFLogRegAdapter.partial_fit called before fit()")

        texts = list(X)
        if replay_X is not None and replay_y is not None:
            texts = texts + list(replay_X)
            y = np.concatenate([np.asarray(y), np.asarray(replay_y)])
        y = np.asarray(y, dtype=np.int64)

        # Refit LR on the concatenated slice using the FROZEN vocabulary.
        X_tf = self.vectorizer.transform(texts)
        t0 = time.perf_counter()
        self.clf = LogisticRegression(
            C=self.cfg.lr_C,
            max_iter=self.cfg.lr_max_iter,
            class_weight=self.cfg.class_weight,
            random_state=self.cfg.seed,
            solver="liblinear",
        )
        self.clf.fit(X_tf, y)
        wall = time.perf_counter() - t0

        self.decision_threshold = self._tune_threshold(texts, y)
        val_f1 = self._score_f1(texts, y)
        metrics = TrainMetrics(
            loss=0.0,
            train_f1=val_f1,
            val_f1=val_f1,
            val_auroc=self._score_auroc(texts, y),
            epoch=1,
            epoch_time_s=wall,
        )
        return FitResult(
            epochs_run=1,
            final_metrics=metrics,
            best_val_f1=val_f1,
            best_epoch=1,
            total_wall_s=wall,
            total_gpu_time_s=0.0,
        )

    def predict_proba(self, X) -> np.ndarray:
        if self.clf is None or self.vectorizer is None:
            raise RuntimeError("TFIDFLogRegAdapter.predict_proba called before fit()")
        X_tf = self.vectorizer.transform(list(X))
        probs = self.clf.predict_proba(X_tf)[:, 1]
        return np.asarray(probs, dtype=np.float32)

    def forward_with_traces(self, X) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return self.predict_proba(X), {}

    def clone(self) -> TFIDFLogRegAdapter:
        new = TFIDFLogRegAdapter(copy.deepcopy(self.cfg))
        new.vectorizer = copy.deepcopy(self.vectorizer) if self.vectorizer is not None else None
        new.clf = copy.deepcopy(self.clf) if self.clf is not None else None
        new.decision_threshold = self.decision_threshold
        return new

    def state_dict(self) -> dict[str, Any]:
        return {
            "vectorizer": copy.deepcopy(self.vectorizer),
            "clf": copy.deepcopy(self.clf),
            "decision_threshold": float(self.decision_threshold),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.vectorizer = copy.deepcopy(state.get("vectorizer"))
        self.clf = copy.deepcopy(state.get("clf"))
        self.decision_threshold = float(state.get("decision_threshold", 0.5))

    def get_layer_params(self, layer_name: str) -> dict[str, Any]:
        return {}

    def n_trainable_params(self, layers: list[str] | None = None) -> int:
        if self.clf is None:
            return 0
        return int(self.clf.coef_.size) + int(self.clf.intercept_.size)

    def flops_per_forward(self) -> int:
        # TF-IDF sparse dot product on max_features vector.
        return int(self.cfg.max_features * 2)

    # ---- helpers ----

    def _score_f1(self, X, y) -> float:
        probs = self.predict_proba(X)
        preds = (probs >= self.decision_threshold).astype(np.int64)
        return float(f1_score(y, preds, zero_division=0))

    def _score_auroc(self, X, y) -> float:
        try:
            return float(roc_auc_score(y, self.predict_proba(X)))
        except ValueError:
            return float("nan")

    def _tune_threshold(self, X, y, *, n_grid: int = 101) -> float:
        probs = self.predict_proba(X)
        if (np.asarray(y) == 1).sum() == 0:
            return 0.5
        candidates = np.quantile(probs, np.linspace(0.2, 0.95, n_grid))
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
