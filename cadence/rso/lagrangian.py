"""Augmented-Lagrangian wrapper for the RSO sandbox environment.

Fixes W-22. The original R-4 setup used a fixed penalty coefficient
`λ_sla` in the reward. That gives PPO no feedback about *how badly* it is
violating the SLA — the same +/-λ term whether it just barely misses the
target or misses by half.

Standard fix for constrained RL: treat λ as a dual variable and update it
between episodes via projected gradient ascent on the constraint violation.

Concretely, after episode k with mean SLA violation `g_k = mean(max(0, sla - post_f1))`,
    λ_{k+1} = max(λ_min, λ_k + η · (g_k - target_violation))

Where `target_violation` is a small slack (0 by default = strict). If the
policy is over-violating, λ grows and pushes PPO toward more retraining; if
the policy is over-satisfying (g_k < 0 → clamped to 0), λ decays back to a
floor so PPO can save cost.

The wrapper is a `gymnasium.Wrapper` — it does not change the observation
space, action space, or the underlying `RetrainingSandboxEnv`'s per-step
mechanics. It only overrides the SLA penalty coefficient by rewriting the
env's `cfg.lambda_sla` at reset() boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from cadence.common.logging import get_logger
from cadence.rso.env import RetrainingSandboxEnv

log = get_logger("cadence.rso.lagrangian")


@dataclass
class AugmentedLagrangianConfig:
    """Dual-update controls."""

    lambda_init: float = 1.0
    lambda_min: float = 0.0
    lambda_max: float = 100.0
    dual_lr: float = 5.0
    # target_violation > 0 = tolerate that much average SLA slack before
    # tightening. 0 = strict.
    target_violation: float = 0.0
    # Exponential moving-average smoothing on the per-episode violation so
    # single outliers don't spike λ.
    ema_beta: float = 0.5


class AugmentedLagrangianEnv(gym.Wrapper):
    """Wrap a sandbox env and update its lambda_sla between episodes.

    Usage:
        env = RetrainingSandboxEnv(...)
        env = AugmentedLagrangianEnv(env, cfg=AugmentedLagrangianConfig())
        model = PPO("MlpPolicy", env, ...).learn(...)

    After training, `env.lambda_history` returns the trajectory of dual
    updates for the R-Gate-A results.md write-up.
    """

    def __init__(
        self,
        env: RetrainingSandboxEnv,
        *,
        cfg: AugmentedLagrangianConfig | None = None,
    ) -> None:
        super().__init__(env)
        self._cfg = cfg or AugmentedLagrangianConfig()
        self._lambda = float(self._cfg.lambda_init)
        self._episode_violations: list[float] = []
        self._ema_violation: float = 0.0
        self.lambda_history: list[float] = [self._lambda]
        # Apply the initial value so the very first episode uses it.
        env.cfg.lambda_sla = self._lambda

    @property
    def current_lambda(self) -> float:
        return self._lambda

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # If a previous episode ran, update lambda based on its mean violation.
        if self._episode_violations:
            mean_v = float(np.mean(self._episode_violations))
            self._ema_violation = (
                self._cfg.ema_beta * self._ema_violation
                + (1 - self._cfg.ema_beta) * mean_v
            )
            # Dual gradient step.
            grad = self._ema_violation - self._cfg.target_violation
            self._lambda = float(
                np.clip(self._lambda + self._cfg.dual_lr * grad, self._cfg.lambda_min, self._cfg.lambda_max)
            )
            self.env.cfg.lambda_sla = self._lambda
            self.lambda_history.append(self._lambda)
            log.info(
                "dual_update",
                lambda_new=self._lambda,
                episode_mean_violation=mean_v,
                ema_violation=self._ema_violation,
            )
        self._episode_violations = []
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        post_f1 = float(info.get("post_f1", 0.0))
        violation = max(0.0, self.env.cfg.sla_target - post_f1)
        self._episode_violations.append(violation)
        info = dict(info)
        info["lambda_sla"] = self._lambda
        info["sla_violation"] = violation
        return obs, reward, terminated, truncated, info
