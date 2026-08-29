"""Integration test: RetrainingSandboxEnv satisfies the gymnasium contract
and produces a reasonable reward decomposition.

Uses a tiny synthetic model + synthetic data so the test runs in <5s on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from benchmarks.synthetic_drift_gen import build_default_scenarios
from cadence.adapters.neural import FraudNet, FraudNetConfig
from cadence.collector.drift_trigger import DriftTriggerConfig, PSITrigger
from cadence.common.seeds import set_global_seed
from cadence.rso.env import RetrainingSandboxEnv, SandboxConfig
from cadence.rso.scorers import PSIResponsibilityScorer


@pytest.fixture
def tiny_env():
    set_global_seed(0, deterministic_torch=False)
    n_feat = 8
    n_train = 400
    X = np.random.randn(n_train, n_feat).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)  # simple learnable signal

    adapter = FraudNet(
        FraudNetConfig(
            input_dim=n_feat, hidden_dims=[16, 8], max_epochs=2, early_stopping_patience=1
        )
    )
    adapter.fit(X, y, X_val=X[:100], y_val=y[:100])

    trigger = PSITrigger(
        baseline_X=X,
        feature_names=[f"f{i}" for i in range(n_feat)],
        cfg=DriftTriggerConfig(psi_threshold=0.1, min_window_rows=50),
    )
    scorer = PSIResponsibilityScorer(trigger=trigger)

    scenarios = build_default_scenarios([f"f{i}" for i in range(n_feat)])
    scenario = scenarios[0]

    # Apply drift to stream once so all rows are drifted (matches phase1_run).
    from benchmarks.synthetic_drift_gen import DriftInjector

    inj = DriftInjector(scenario, rng=np.random.default_rng(1))
    stream_X, stream_y = inj.apply(X.copy(), y.copy(), step_offset=0)

    env = RetrainingSandboxEnv(
        initial_adapter=adapter,
        historical_X=X,
        historical_y=y,
        drifted_stream_X=stream_X,
        drifted_stream_y=stream_y,
        responsibility_scorer=scorer,
        cfg=SandboxConfig(
            window_size=200,
            max_windows_per_episode=2,
            finetune_epochs=1,
            fullretrain_epochs=1,
            sla_target=0.5,
        ),
    )
    return env


def test_env_passes_sb3_check(tiny_env):
    # skip_render_check because we don't implement rendering.
    check_env(tiny_env, warn=True, skip_render_check=True)


def test_env_reset_returns_valid_observation(tiny_env):
    obs, info = tiny_env.reset(seed=0)
    assert tiny_env.observation_space.contains(obs)
    assert isinstance(info, dict)


def test_env_step_returns_valid_transition(tiny_env):
    tiny_env.reset(seed=0)
    obs, reward, terminated, truncated, info = tiny_env.step(0)  # no-op
    assert tiny_env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "post_f1" in info and "cost_gpu_hr" in info


def test_env_step_full_action_costs_more_than_partial(tiny_env):
    """Full retrain must be more expensive than partial (D-7 cost model)."""
    tiny_env.reset(seed=0)
    _, _, _, _, info_partial = tiny_env.step(1)
    tiny_env.reset(seed=0)
    _, _, _, _, info_full = tiny_env.step(2)
    assert info_full["cost_gpu_hr"] > info_partial["cost_gpu_hr"]
    assert info_full["cost_kg_co2"] > info_partial["cost_kg_co2"]


def test_env_step_noop_is_free(tiny_env):
    tiny_env.reset(seed=0)
    _, _, _, _, info = tiny_env.step(0)
    assert info["cost_gpu_hr"] == 0.0
    assert info["cost_kg_co2"] == 0.0
