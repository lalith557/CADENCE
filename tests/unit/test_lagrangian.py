"""Guards for the Augmented-Lagrangian wrapper."""

from __future__ import annotations

import numpy as np

from cadence.rso.env import RetrainingSandboxEnv, SandboxConfig
from cadence.rso.lagrangian import AugmentedLagrangianConfig, AugmentedLagrangianEnv


class _StubAdapter:
    """Minimal adapter shim so we can construct the env without training."""

    def __init__(self, n_features: int) -> None:
        self.n_features = n_features
        self.decision_threshold = 0.5

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def predict_proba(self, X):
        # Deterministic near-zero probs -> F1 ~ 0 -> SLA violation.
        return np.full(X.shape[0], 0.01, dtype=np.float32)

    def fit(self, X, y, **kw):
        pass

    def partial_fit(self, X, y, **kw):
        pass

    def flops_per_forward(self) -> int:
        return 1_000

    def clone(self):
        return _StubAdapter(self.n_features)


def _make_env(sla: float = 0.9) -> RetrainingSandboxEnv:
    n = 8
    rng = np.random.default_rng(0)
    Xh = rng.standard_normal((256, n)).astype(np.float32)
    yh = (rng.random(256) < 0.1).astype(np.int64)
    Xs = rng.standard_normal((512, n)).astype(np.float32)
    ys = (rng.random(512) < 0.1).astype(np.int64)

    def scorer(adapter, X, y):
        return rng.random(n).astype(np.float32)

    return RetrainingSandboxEnv(
        initial_adapter=_StubAdapter(n),
        historical_X=Xh,
        historical_y=yh,
        drifted_stream_X=Xs,
        drifted_stream_y=ys,
        responsibility_scorer=scorer,
        cfg=SandboxConfig(
            window_size=64, max_windows_per_episode=4, sla_target=sla, ewc_penalty=0.0
        ),
    )


def test_dual_lambda_grows_when_policy_violates_sla() -> None:
    env = _make_env(sla=0.9)
    wrapped = AugmentedLagrangianEnv(
        env, cfg=AugmentedLagrangianConfig(lambda_init=1.0, dual_lr=2.0, ema_beta=0.0)
    )
    # First episode — do nothing, F1 stays 0, violation ~= 0.9.
    obs, _ = wrapped.reset()
    lambda0 = wrapped.current_lambda
    for _ in range(4):
        obs, _, term, trunc, _ = wrapped.step(0)
        if term or trunc:
            break
    # Trigger dual update by starting a new episode.
    wrapped.reset()
    lambda1 = wrapped.current_lambda
    assert lambda1 > lambda0, f"expected lambda to grow, got {lambda0} -> {lambda1}"


def test_dual_lambda_never_exceeds_max_or_drops_below_min() -> None:
    env = _make_env()
    wrapped = AugmentedLagrangianEnv(
        env, cfg=AugmentedLagrangianConfig(lambda_init=50.0, lambda_max=60.0, lambda_min=1.0, dual_lr=10_000.0)
    )
    for _ in range(3):
        wrapped.reset()
        for _ in range(4):
            _, _, term, trunc, _ = wrapped.step(0)
            if term or trunc:
                break
    assert wrapped.current_lambda <= 60.0
    assert wrapped.current_lambda >= 1.0


def test_info_dict_reports_lambda_and_violation() -> None:
    env = _make_env()
    wrapped = AugmentedLagrangianEnv(env, cfg=AugmentedLagrangianConfig())
    wrapped.reset()
    _, _, _, _, info = wrapped.step(0)
    assert "lambda_sla" in info
    assert "sla_violation" in info
    assert info["sla_violation"] >= 0.0
