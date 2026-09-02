"""Regression tests for the Gate-A pre-fix audit (see docs/gate_a_audit.md).

Every test here starts as a *failing* test that pins one confirmed bug.
The corresponding fix (W-32..W-37) makes the test pass. Keep the tests
even after the fixes land — they're the paper's guarantees that the
Gate-A pipeline stays honest.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cadence.carbon.model import GridProfile, HardwareProfile, estimate_cost
from cadence.rso.env import RetrainingSandboxEnv, SandboxConfig


# ---------- W-32: reward scaling ----------


class _StubAdapter:
    """Minimal adapter shim — no training, deterministic F1 outputs."""

    family = "stub"

    def __init__(self, n_features: int = 4, f1: float = 0.5) -> None:
        self._f1 = f1
        self.decision_threshold = 0.5
        self._n_features = n_features

    def state_dict(self) -> dict:
        return {"f1": self._f1}

    def load_state_dict(self, state: dict) -> None:
        self._f1 = state.get("f1", self._f1)

    def predict_proba(self, X):
        # Return probabilities that yield the configured F1 on any y.
        n = X.shape[0]
        return np.full(n, self._f1, dtype=np.float32)

    def flops_per_forward(self) -> int:
        return 4100  # FraudNet-scale

    def fit(self, X, y, **_):
        return None

    def partial_fit(self, X, y, **_):
        return None

    def clone(self) -> "_StubAdapter":
        return _StubAdapter(self._n_features, self._f1)


def _dummy_env(cfg: SandboxConfig, adapter: _StubAdapter | None = None) -> RetrainingSandboxEnv:
    if adapter is None:
        adapter = _StubAdapter()
    rng = np.random.default_rng(0)
    hist_X = rng.standard_normal((256, adapter._n_features)).astype(np.float32)
    hist_y = (rng.random(256) < 0.3).astype(np.int64)
    stream_X = rng.standard_normal((2048, adapter._n_features)).astype(np.float32)
    stream_y = (rng.random(2048) < 0.3).astype(np.int64)

    def scorer(a, X, y):
        return rng.random(a._n_features).astype(np.float32)

    return RetrainingSandboxEnv(
        initial_adapter=adapter,
        historical_X=hist_X,
        historical_y=hist_y,
        drifted_stream_X=stream_X,
        drifted_stream_y=stream_y,
        responsibility_scorer=scorer,
        cfg=cfg,
    )


def test_reward_cost_term_is_at_least_1pct_of_delta_f1_for_full_retrain() -> None:
    """W-32: after the reward-scaling fix, the cost penalty for a full
    retrain must be non-negligible relative to Δf1.

    Concretely: for a full retrain with default finetune/fullretrain epochs
    and default reward weights, |w_gpu_hr · gpu_hr + w_kg_co2 · kg_co2|
    should be at least 1% of a 0.05 F1-point Δ. Prior code has it at
    ~1.5e-7 which is 10⁶× too small.
    """
    cfg = SandboxConfig(
        window_size=2048,
        fullretrain_epochs=15,
        w_gpu_hr=0.05,
        w_kg_co2=0.5,
    )
    cost = estimate_cost(
        flops_per_forward=4100,
        n_samples=22048,
        epochs=cfg.fullretrain_epochs,
        hardware=cfg.hardware,
        grid=cfg.grid,
    )
    gpu_hr = cost.gpu_seconds / 3600.0
    kg_co2 = cost.kg_co2
    # The APPLIED reward penalty (what env.step subtracts).
    penalty = cfg.w_gpu_hr * gpu_hr + cfg.w_kg_co2 * kg_co2

    # Apply the scaling knobs the fix should add.
    penalty_scaled = penalty * getattr(cfg, "cost_scale", 1.0)

    delta_f1_example = 0.05
    ratio = penalty_scaled / delta_f1_example
    # W-32 fix threshold: cost must be at least 1% of a 0.05 Δf1 -> 5e-4.
    assert ratio >= 0.01, (
        f"W-32 regression: cost penalty is {penalty_scaled:.3e}, only "
        f"{ratio * 100:.4f}% of a 0.05 delta_f1. Cost signal is invisible "
        f"to PPO. Fix by setting SandboxConfig.cost_scale so full-retrain "
        f"cost lands around 0.001-0.01 (comparable to Δf1)."
    )


# ---------- W-33: ent_coef ----------


def test_ppo_train_config_exposes_ent_coef() -> None:
    """W-33: ent_coef=0 causes deterministic collapse. PPOTrainConfig
    must expose ent_coef as a field so callers can prevent it.
    """
    from cadence.rso.ppo import PPOTrainConfig

    cfg = PPOTrainConfig()
    assert hasattr(cfg, "ent_coef"), (
        "W-33 regression: PPOTrainConfig has no ent_coef field. SB3's "
        "default ent_coef=0 collapses PPO to a deterministic policy on "
        "Discrete(3). Add ent_coef with a nonzero default (e.g. 0.01)."
    )
    assert cfg.ent_coef > 0.0, (
        f"W-33 regression: PPOTrainConfig.ent_coef={cfg.ent_coef}. "
        f"Zero entropy on a tiny action space collapses deterministically."
    )


# ---------- W-34: VecNormalize ----------


def test_train_ppo_wraps_env_in_vec_normalize_when_requested() -> None:
    """W-34: PPO's MLP takes raw observations. The obs vector spans
    [0, 10000]. Without VecNormalize the network's early gradient
    dynamics are dominated by scale, not information.
    """
    from cadence.rso.ppo import PPOTrainConfig, train_ppo

    def _factory():
        return _dummy_env(SandboxConfig(window_size=64, max_windows_per_episode=3))

    # train_ppo(normalize=True) should return a model whose env stack
    # includes a VecNormalize.
    try:
        model = train_ppo(
            _factory,
            cfg=PPOTrainConfig(total_timesteps=64, n_steps=32, batch_size=16, n_epochs=1),
            check_env_first=False,
            normalize=True,
        )
    except TypeError as e:
        pytest.fail(
            f"W-34 regression: train_ppo has no `normalize` kwarg. Add it "
            f"and thread it to a VecNormalize wrapper. Underlying error: {e}"
        )
    from stable_baselines3.common.vec_env import VecNormalize

    assert isinstance(model.env, VecNormalize) or isinstance(
        getattr(model.env, "venv", None), VecNormalize
    ), "W-34 regression: model env stack does not contain VecNormalize"


# ---------- W-35: prequential training reward ----------


def test_train_reward_is_prequential_not_in_sample() -> None:
    """W-35: env.step retrained on window t and evaluated F1 on window t
    (in-sample). Prequential means F1 is measured on window t+1.
    """
    cfg = SandboxConfig(window_size=64, max_windows_per_episode=5)
    env = _dummy_env(cfg)
    env.reset()
    _, _, _, _, info = env.step(0)  # no-op
    # In-sample: post_f1 comes from the same window as pre_f1. Prequential:
    # post_f1 comes from the next window (or is documented as the "reward
    # window" separate from the "eval window").
    assert "reward_window_f1" in info or info.get("prequential"), (
        "W-35 regression: env.step info lacks a `reward_window_f1` field "
        "or a `prequential=True` flag. Training reward is in-sample "
        "(same window as retrain fit); eval reports prequential. PPO is "
        "optimizing an inflated reward it never faces at eval."
    )


# ---------- W-36: per-seed policies ----------


def test_phase_a_run_supports_per_seed_ppo_training() -> None:
    """W-36: phase_a_run.py currently trains PPO once (seed=42) OUTSIDE
    the seed loop. The loop only varies eval-stream sampling. This is
    pseudoreplication; every "seed" uses the same policy weights.

    The fix is to move the train_ppo call INSIDE the seed loop and pass
    PPOTrainConfig(seed=seed) so each seed gets an independent policy.
    Verified here by importing the module and looking for a callable
    that trains once per seed.
    """
    import inspect

    from benchmarks import phase_a_run

    src = inspect.getsource(phase_a_run.main)
    # Post-fix: the module should either train inside the seed loop, or
    # expose a `--per-seed-policies` flag, or set the train call's seed
    # from the loop index. We detect any of those via a substring check.
    has_per_seed = (
        "per_seed_policies" in src
        or "for seed in range(args.seeds)" in src and "train_ppo" in src.split("for seed in range(args.seeds)")[1]
    )
    assert has_per_seed, (
        "W-36 regression: phase_a_run.main trains PPO once outside the seed "
        "loop. Move the train_ppo call inside `for seed in range(args.seeds)` "
        "with PPOTrainConfig(seed=seed) to make the reported n=10 samples "
        "actually independent."
    )


# ---------- W-37: replay RNG per env instance ----------


def test_step_info_exposes_reward_components_for_stage1_gates() -> None:
    """Stage-1 gates G3/G4/G5 need the reward decomposition per step.

    env.step must return info with:
      * delta_f1_component
      * cost_penalty_component
      * sla_penalty_component
    that sum (with signs) to the scalar reward.
    """
    cfg = SandboxConfig(window_size=64, max_windows_per_episode=5, cost_scale=1e7)
    env = _dummy_env(cfg)
    env.reset()
    _, reward, _, _, info = env.step(2)  # full retrain — nonzero cost
    for key in ("delta_f1_component", "cost_penalty_component", "sla_penalty_component"):
        assert key in info, (
            f"Stage-1 regression: env.step info lacks `{key}`. Gates G3/G4/G5 "
            f"need to see the reward decomposition per step; add these to the "
            f"info dict in env.step (they cost nothing extra to compute)."
        )
    # Round-trip: components sum (with signs) to reward.
    reconstructed = (
        info["delta_f1_component"]
        - info["cost_penalty_component"]
        - info["sla_penalty_component"]
    )
    assert abs(reconstructed - reward) < 1e-6, (
        f"Stage-1 regression: reward components don't sum to reward "
        f"({reconstructed} vs {reward}). The Stage-1 sanity gates rely on "
        f"decomposing the scalar reward into these three."
    )


def test_replay_batch_differs_across_consecutive_partial_retrains() -> None:
    """W-37: env._do_partial_retrain uses np.random.default_rng(0) freshly
    every call. Two consecutive calls select identical replay indices.

    We patch adapter.partial_fit to capture the replay_X id() and assert
    two consecutive calls receive DIFFERENT arrays.
    """
    seen_replay_ids: list[int] = []

    class _RecordingAdapter(_StubAdapter):
        def partial_fit(self, X, y, *, replay_X=None, **kw):
            if replay_X is not None:
                # Hash the first-row bytes to identify uniqueness.
                seen_replay_ids.append(int(replay_X[0].sum() * 1e6))
            return None

    cfg = SandboxConfig(
        window_size=64,
        max_windows_per_episode=5,
        replay_buffer_size=100,
        replay_ratio_new=0.5,
        ewc_penalty=0.0,  # disable EWC path so partial_fit is called cleanly
    )
    env = _dummy_env(cfg, adapter=_RecordingAdapter())
    env.reset()
    env.step(1)  # partial
    env.step(1)  # partial
    assert len(seen_replay_ids) == 2, "two partial calls should have run"
    assert seen_replay_ids[0] != seen_replay_ids[1], (
        f"W-37 regression: replay batch identical across consecutive partial "
        f"retrains ({seen_replay_ids}). Fix by storing self._replay_rng on "
        f"env init and using it in _do_partial_retrain."
    )
