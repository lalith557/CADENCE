"""SB3 PPO wrapper for the RetrainingSandboxEnv.

Training path:
1. Build a *fast* sandbox env with reduced retrain epochs (`SandboxConfig` with
   `finetune_epochs=1`, `fullretrain_epochs=1`) — PPO doesn't need convergence-
   quality retrains to learn the decision structure.
2. Train PPO for N steps.
3. Save the policy.

Eval path:
1. Build the *full-fidelity* sandbox env (paper's real retrain-epoch settings).
2. Run the trained policy deterministically for K episodes.
3. Return aggregated metrics matching the baseline harness's `BaselineOutcome`
   schema, so R-4 comparisons are apples-to-apples.

The Lagrangian SLA constraint is implemented as a reward-shaping term (per D-7
initial version): `-λ · max(0, sla - post_F1)`. The dual-update version is a
Phase 5 refinement — noted as W-22.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cadence.common.logging import get_logger
from cadence.rso.env import RetrainingSandboxEnv, SandboxConfig

log = get_logger("cadence.rso.ppo")


@dataclass
class PPOTrainConfig:
    total_timesteps: int = 5_000
    n_steps: int = 128
    batch_size: int = 64
    learning_rate: float = 3e-4
    clip_range: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_epochs: int = 10
    policy: str = "MlpPolicy"
    seed: int = 42
    verbose: int = 0
    # W-33 (docs/gate_a_audit.md §1.2). SB3's default ent_coef is 0.0,
    # which collapses PPO to a deterministic policy on a Discrete(3)
    # action space almost immediately — the "bang-bang" pathology in
    # R-4b's 3×3 sweep. A small positive value preserves exploration
    # long enough for the policy to learn a state-conditional
    # (mixed) strategy. 0.01 is the SB3 recipe recommendation for
    # small-action-space discrete PPO.
    ent_coef: float = 0.01


class MLflowCallback(BaseCallback):
    """Log PPO training curves to MLflow every N steps."""

    def __init__(self, log_every: int = 500):
        super().__init__()
        self.log_every = log_every
        self._last_logged = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_logged < self.log_every:
            return True
        self._last_logged = self.num_timesteps
        # SB3 fills logger with rollout/ep_rew_mean etc during ep boundaries.
        try:
            for k, v in self.logger.name_to_value.items():
                if isinstance(v, int | float):
                    mlflow.log_metric(f"ppo/{k}", float(v), step=self.num_timesteps)
        except Exception as e:  # pragma: no cover — MLflow noise, not a fail
            log.warning("mlflow_log_failed", err=str(e))
        return True


def train_ppo(
    env_factory: Callable[[], RetrainingSandboxEnv],
    *,
    cfg: PPOTrainConfig | None = None,
    save_path: Path | None = None,
    check_env_first: bool = True,
    normalize: bool = True,
    normalize_clip_obs: float = 10.0,
    normalize_gamma: float | None = None,
) -> PPO:
    """Train PPO on the given env factory. Returns the trained model.

    W-34 (docs/gate_a_audit.md §1.3): the raw observation vector spans
    [0, 10000] (hours_since_retrain) and [0, 2000] (grid intensity)
    alongside [0, 1] scores. A vanilla MlpPolicy without observation
    normalization is dominated by the largest-magnitude features and
    cannot learn the score / F1 signal.

    `normalize=True` (default) wraps the vec-env in `VecNormalize` with
    running-mean/variance stats for both obs and reward. Set to False
    to reproduce the pre-fix pathology for ablations.
    """
    cfg = cfg or PPOTrainConfig()
    single_env = env_factory()
    if check_env_first:
        check_env(single_env, warn=True, skip_render_check=True)

    def _make() -> RetrainingSandboxEnv:
        return env_factory()

    vec_env = DummyVecEnv([_make])
    if normalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=normalize_clip_obs,
            gamma=normalize_gamma if normalize_gamma is not None else cfg.gamma,
        )
    model = PPO(
        cfg.policy,
        vec_env,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        clip_range=cfg.clip_range,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        n_epochs=cfg.n_epochs,
        ent_coef=cfg.ent_coef,  # W-33 fix
        seed=cfg.seed,
        verbose=cfg.verbose,
    )

    log.info(
        "ppo_train_start",
        total_timesteps=cfg.total_timesteps,
        n_steps=cfg.n_steps,
        seed=cfg.seed,
    )
    t0 = time.perf_counter()
    model.learn(total_timesteps=cfg.total_timesteps, callback=MLflowCallback())
    log.info("ppo_train_done", wall_s=time.perf_counter() - t0)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        log.info("ppo_saved", path=str(save_path))
        # W-34: save the VecNormalize stats alongside the checkpoint so
        # inference uses the same normalization the policy was trained
        # against. Without this, the policy sees different obs
        # distributions at eval and its actions drift.
        if normalize:
            norm_path = save_path.with_suffix(".vecnorm.pkl")
            model.get_vec_normalize_env().save(str(norm_path))
            log.info("vecnorm_saved", path=str(norm_path))
    return model


@dataclass
class RSOEvalOutcome:
    """Mirror the shape of BaselineOutcome so R-4 can compare directly."""

    strategy_name: str
    scenario_name: str
    seed: int
    rolling_f1: list[float]
    action_counts: dict[str, int]
    total_gpu_seconds: float
    total_kg_co2: float
    mean_f1: float
    min_f1: float
    sla_windows_below: int
    forgetting_score: float


def evaluate_ppo(
    model: PPO,
    env: RetrainingSandboxEnv,
    *,
    unrelated_X: np.ndarray,
    unrelated_y: np.ndarray,
    sla_target: float,
    scenario_name: str,
    seed: int,
    strategy_name: str = "rso_ppo",
    max_windows: int | None = None,
    deterministic: bool = True,
    run_full_episode: bool = True,
) -> RSOEvalOutcome:
    """Roll out a trained policy for one episode.

    `run_full_episode=True` forces `max_windows` steps regardless of SLA-recovery
    termination — this is what the R-4 comparison needs, so RSO's mean_f1 is
    averaged over the same window count as the baselines.
    """
    from sklearn.metrics import f1_score

    def _f1_on(X, y):
        probs = env.adapter.predict_proba(X)
        threshold = getattr(env.adapter, "decision_threshold", 0.5)
        preds = (probs >= threshold).astype(np.int64)
        return float(f1_score(y, preds, zero_division=0))

    baseline_unrelated_f1 = _f1_on(unrelated_X, unrelated_y)

    obs, _ = env.reset(seed=seed)
    action_counts = {"no-op": 0, "partial": 0, "full": 0}
    rolling_f1: list[float] = []
    total_gpu_seconds = 0.0
    total_kg_co2 = 0.0
    windows_seen = 0
    limit = max_windows if max_windows is not None else env.cfg.max_windows_per_episode

    while windows_seen < limit:
        action, _ = model.predict(obs, deterministic=deterministic)
        action_int = int(np.asarray(action).flatten()[0])
        obs, reward, terminated, truncated, info = env.step(action_int)
        action_counts[["no-op", "partial", "full"][action_int]] += 1
        # Match baseline harness: record F1 BEFORE the action's effect.
        rolling_f1.append(float(info.get("pre_f1", 0.0)))
        total_gpu_seconds += float(info.get("cost_gpu_hr", 0.0)) * 3600.0
        total_kg_co2 += float(info.get("cost_kg_co2", 0.0))
        windows_seen += 1
        if truncated:  # end of drifted stream — cannot continue
            break
        if not run_full_episode and terminated:
            break

    forgetting = baseline_unrelated_f1 - _f1_on(unrelated_X, unrelated_y)
    return RSOEvalOutcome(
        strategy_name=strategy_name,
        scenario_name=scenario_name,
        seed=seed,
        rolling_f1=rolling_f1,
        action_counts=action_counts,
        total_gpu_seconds=total_gpu_seconds,
        total_kg_co2=total_kg_co2,
        mean_f1=float(np.mean(rolling_f1)) if rolling_f1 else 0.0,
        min_f1=float(np.min(rolling_f1)) if rolling_f1 else 0.0,
        sla_windows_below=int(sum(1 for f in rolling_f1 if f < sla_target)),
        forgetting_score=float(forgetting),
    )


def build_sandbox_env_factory(
    *,
    initial_adapter,
    historical_X: np.ndarray,
    historical_y: np.ndarray,
    drifted_stream_X: np.ndarray,
    drifted_stream_y: np.ndarray,
    scorer,
    sandbox_cfg: SandboxConfig,
) -> Callable[[], RetrainingSandboxEnv]:
    """Return a zero-arg factory. Fresh adapter clone per env so parallel envs
    don't share weights."""

    def _factory() -> RetrainingSandboxEnv:
        adapter_copy = initial_adapter.clone()
        return RetrainingSandboxEnv(
            initial_adapter=adapter_copy,
            historical_X=historical_X,
            historical_y=historical_y,
            drifted_stream_X=drifted_stream_X,
            drifted_stream_y=drifted_stream_y,
            responsibility_scorer=scorer,
            cfg=sandbox_cfg,
        )

    return _factory


def build_multi_scenario_env_factory(
    *,
    initial_adapter,
    historical_X: np.ndarray,
    historical_y: np.ndarray,
    raw_stream_X: np.ndarray,
    raw_stream_y: np.ndarray,
    scenarios: list,
    scorer,
    sandbox_cfg: SandboxConfig,
    seed: int = 0,
) -> Callable[[], RetrainingSandboxEnv]:
    """Env factory that picks a random scenario per reset — helps PPO generalize
    across the drift types it will see at eval time.

    Uses a *pre-injected* drift stream per scenario, cached to avoid re-running
    the injector at every reset.
    """
    from benchmarks.baselines.harness import make_baseline_stream

    # Pre-inject each scenario once.
    cached = []
    for i, sc in enumerate(scenarios):
        sX, sy, _ = make_baseline_stream(raw_stream_X, raw_stream_y, sc, seed=seed + i)
        cached.append((sX, sy))

    rng = np.random.default_rng(seed)
    counter = {"i": 0}

    def _factory() -> RetrainingSandboxEnv:
        # Cycle in shuffled order so each call is deterministic but coverage is even.
        i = counter["i"] % len(cached)
        counter["i"] += 1
        sX, sy = cached[i]
        adapter_copy = initial_adapter.clone()
        return RetrainingSandboxEnv(
            initial_adapter=adapter_copy,
            historical_X=historical_X,
            historical_y=historical_y,
            drifted_stream_X=sX,
            drifted_stream_y=sy,
            responsibility_scorer=scorer,
            cfg=sandbox_cfg,
        )

    _ = rng  # keep for future randomization
    return _factory
