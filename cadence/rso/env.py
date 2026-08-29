"""RetrainingSandboxEnv — the Gym environment PPO trains against.

State (13-d per D-7):
  0..4  top-5 responsibility scores (sorted desc, padded with 0 if fewer)
  5     current rolling F1
  6     SLA target
  7     drift duration in units of "windows since alert fired"
  8..10 per-action GPU-hr estimate (no-op = 0)
  11    current grid gCO2 per kWh
  12    hours since last retrain

Action (Discrete(3)):
  0 no-op
  1 partial retrain (uses the top-responsibility layer / D-8 mechanics)
  2 full retrain

Reward = (post-action F1 - pre-action F1)
       - w1 · gpu_hr
       - w2 · kg_co2
       - λ · max(0, sla - post_action_F1)

The env owns:
  * a copy of the production model (adapter)
  * a stream of drifted data + the ground-truth injector
  * a running responsibility oracle (provided externally — for Phase 1 baselines
    we use a rule-based stub so H2 numbers can land before Phase 3's CDAG exists)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cadence.adapters.base import ModelAdapter
from cadence.carbon.model import GridProfile, HardwareProfile, estimate_cost
from cadence.common.logging import get_logger

log = get_logger("cadence.rso.env")


# Type: given (adapter, X_window, y_window), return length-N vector of
# per-feature responsibility scores in [0, 1].
ResponsibilityScorer = Callable[[ModelAdapter, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class SandboxConfig:
    """RSO sandbox settings."""

    window_size: int = 2048
    max_windows_per_episode: int = 20
    top_k: int = 5
    w_gpu_hr: float = 0.05
    w_kg_co2: float = 0.5
    lambda_sla: float = 1.0
    sla_target: float = 0.90
    finetune_epochs: int = 5
    fullretrain_epochs: int = 15
    replay_buffer_size: int = 5000
    replay_ratio_new: float = 0.7
    hardware: HardwareProfile = field(default_factory=HardwareProfile)
    grid: GridProfile = field(default_factory=GridProfile)


class RetrainingSandboxEnv(gym.Env):
    """A single-episode = single drift scenario. Episode ends after
    `max_windows_per_episode` windows or when F1 fully recovers to SLA."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        initial_adapter: ModelAdapter,
        historical_X: np.ndarray,
        historical_y: np.ndarray,
        drifted_stream_X: np.ndarray,
        drifted_stream_y: np.ndarray,
        responsibility_scorer: ResponsibilityScorer,
        cfg: SandboxConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg or SandboxConfig()
        self._initial_state = initial_adapter.state_dict()
        self.adapter = initial_adapter
        self.historical_X = historical_X
        self.historical_y = historical_y
        self.stream_X = drifted_stream_X
        self.stream_y = drifted_stream_y
        self.scorer = responsibility_scorer

        self.n_features = historical_X.shape[1]
        self.action_space = spaces.Discrete(3)
        # Bounded observation: scores in [0,1], F1 in [0,1], durations bounded.
        self.observation_space = spaces.Box(
            low=np.zeros(13, dtype=np.float32),
            high=np.array(
                [1] * 5 + [1, 1, self.cfg.max_windows_per_episode, 100, 100, 100, 2000, 10_000],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self._cursor = 0
        self._windows_since_alert = 0
        self._hours_since_retrain = 0.0
        self._current_f1 = 0.0
        self._last_window_X: np.ndarray | None = None
        self._last_window_y: np.ndarray | None = None

    # ---- gym API ----

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.adapter.load_state_dict(self._initial_state)
        self._cursor = 0
        self._windows_since_alert = 0
        self._hours_since_retrain = 0.0
        # Peek the first window so reset's F1 matches step 0's expected pre-F1.
        peek_end = min(self.cfg.window_size, self.stream_X.shape[0])
        self._last_window_X = self.stream_X[:peek_end]
        self._last_window_y = self.stream_y[:peek_end]
        self._current_f1 = self._eval_f1_on_last_window()
        return self._observation(scores=np.zeros(self.n_features, dtype=np.float32)), {}

    def step(self, action: int):
        window_X, window_y = self._next_window()
        self._last_window_X = window_X
        self._last_window_y = window_y
        pre_f1 = self._eval_f1_on_last_window()
        cost_gpu_hr = 0.0
        cost_kg_co2 = 0.0

        if action == 0:  # no-op
            pass
        elif action == 1:  # partial
            scores = self.scorer(self.adapter, window_X, window_y)
            top_feature = int(np.argmax(scores))
            # Layer mapping is Adapter-specific; for FraudNet we use "layer1" as
            # the default target because it fires on raw feature statistics.
            target_layer = "layer1"
            cost = estimate_cost(
                flops_per_forward=self.adapter.flops_per_forward(),
                n_samples=window_X.shape[0],
                epochs=self.cfg.finetune_epochs,
                hardware=self.cfg.hardware,
                grid=self.cfg.grid,
            )
            cost_gpu_hr = cost.gpu_seconds / 3600.0
            cost_kg_co2 = cost.kg_co2
            self._do_partial_retrain(window_X, window_y, target_layer=target_layer)
            self._hours_since_retrain = 0.0
            log.info(
                "action_partial",
                target_feature=top_feature,
                target_layer=target_layer,
                **cost.__dict__,
            )
        elif action == 2:  # full
            cost = estimate_cost(
                flops_per_forward=self.adapter.flops_per_forward(),
                n_samples=self.historical_X.shape[0] + window_X.shape[0],
                epochs=self.cfg.fullretrain_epochs,
                hardware=self.cfg.hardware,
                grid=self.cfg.grid,
            )
            cost_gpu_hr = cost.gpu_seconds / 3600.0
            cost_kg_co2 = cost.kg_co2
            self._do_full_retrain(window_X, window_y)
            self._hours_since_retrain = 0.0
            log.info("action_full", **cost.__dict__)

        post_f1 = self._eval_f1_on_last_window()
        self._current_f1 = post_f1
        self._hours_since_retrain += 1.0
        self._windows_since_alert += 1

        # Reward per D-7.
        delta_f1 = post_f1 - pre_f1
        sla_penalty = max(0.0, self.cfg.sla_target - post_f1)
        reward = (
            delta_f1
            - self.cfg.w_gpu_hr * cost_gpu_hr
            - self.cfg.w_kg_co2 * cost_kg_co2
            - self.cfg.lambda_sla * sla_penalty
        )

        terminated = post_f1 >= self.cfg.sla_target
        truncated = self._windows_since_alert >= self.cfg.max_windows_per_episode

        scores = self.scorer(self.adapter, window_X, window_y)
        return (
            self._observation(scores=scores, cost_gpu_hr=cost_gpu_hr),
            float(reward),
            bool(terminated),
            bool(truncated),
            {
                "post_f1": post_f1,
                "pre_f1": pre_f1,
                "cost_gpu_hr": cost_gpu_hr,
                "cost_kg_co2": cost_kg_co2,
            },
        )

    # ---- internals ----

    def _next_window(self) -> tuple[np.ndarray, np.ndarray]:
        end = min(self._cursor + self.cfg.window_size, self.stream_X.shape[0])
        window_X = self.stream_X[self._cursor : end]
        window_y = self.stream_y[self._cursor : end]
        self._cursor = end
        return window_X, window_y

    def _do_full_retrain(self, window_X: np.ndarray, window_y: np.ndarray) -> None:
        X = np.concatenate([self.historical_X, window_X], axis=0)
        y = np.concatenate([self.historical_y, window_y], axis=0)
        self.adapter.fit(X, y)

    def _do_partial_retrain(
        self, window_X: np.ndarray, window_y: np.ndarray, *, target_layer: str
    ) -> None:
        replay_n = int(self.cfg.replay_buffer_size * (1 - self.cfg.replay_ratio_new))
        replay_idx = np.random.default_rng(0).integers(0, self.historical_X.shape[0], size=replay_n)
        replay_X = self.historical_X[replay_idx]
        replay_y = self.historical_y[replay_idx]

        # Delegate to adapter.partial_fit; if the adapter doesn't support it,
        # fall back to full retrain.
        try:
            self.adapter.partial_fit(
                window_X,
                window_y,
                layers_to_update=[target_layer],
                replay_X=replay_X,
                replay_y=replay_y,
                max_epochs=self.cfg.finetune_epochs,
            )
        except NotImplementedError:
            log.warning("adapter_no_partial_fit_fallback_full")
            self._do_full_retrain(window_X, window_y)

    def _eval_f1_on_last_window(self) -> float:
        """Evaluate F1 on the most recently seen window (matches the baseline
        harness's protocol: F1 measured on the same window the strategy just
        acted upon, so mean_f1 comparisons are apples-to-apples)."""
        from sklearn.metrics import f1_score

        if self._last_window_X is None or self._last_window_y is None:
            return 0.0
        probs = self.adapter.predict_proba(self._last_window_X)
        threshold = getattr(self.adapter, "decision_threshold", 0.5)
        preds = (probs >= threshold).astype(np.int64)
        return float(f1_score(self._last_window_y, preds, zero_division=0))

    def _observation(
        self,
        *,
        scores: np.ndarray,
        cost_gpu_hr: float = 0.0,
    ) -> np.ndarray:
        # Top-k scores (padded).
        sorted_scores = np.sort(scores)[::-1]
        top = sorted_scores[: self.cfg.top_k]
        if top.shape[0] < self.cfg.top_k:
            pad = np.zeros(self.cfg.top_k - top.shape[0], dtype=np.float32)
            top = np.concatenate([top, pad])

        # Estimate per-action GPU-hr (no-op / partial / full).
        flops = self.adapter.flops_per_forward()
        partial_cost = estimate_cost(
            flops,
            self.cfg.window_size,
            self.cfg.finetune_epochs,
            hardware=self.cfg.hardware,
            grid=self.cfg.grid,
        )
        full_cost = estimate_cost(
            flops,
            self.historical_X.shape[0] + self.cfg.window_size,
            self.cfg.fullretrain_epochs,
            hardware=self.cfg.hardware,
            grid=self.cfg.grid,
        )

        obs = np.array(
            [
                *top.tolist(),
                float(self._current_f1),
                float(self.cfg.sla_target),
                float(self._windows_since_alert),
                0.0,  # no-op cost
                float(partial_cost.gpu_seconds / 3600.0),
                float(full_cost.gpu_seconds / 3600.0),
                float(self.cfg.grid.intensity_g_per_kwh),
                float(self._hours_since_retrain),
            ],
            dtype=np.float32,
        )
        return obs
