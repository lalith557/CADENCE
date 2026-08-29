"""Shared benchmark harness for Phase 1's three baselines.

Given a trained adapter, a historical dataset, a drift scenario, and a
strategy, simulates one production episode window-by-window and records:
  - F1 across time (rolling)
  - retrain events with their measured cost/carbon
  - end-of-episode aggregates (mean F1, total GPU-hr, total kg CO2)
  - forgetting rate on an unrelated held-out segment (for baseline C)

Deterministic given seed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np
from sklearn.metrics import f1_score

from benchmarks.synthetic_drift_gen import DriftInjector, DriftScenario
from cadence.adapters.base import ModelAdapter
from cadence.carbon.model import GridProfile, HardwareProfile, estimate_cost
from cadence.collector.drift_trigger import DriftTriggerConfig, PSITrigger
from cadence.common.logging import get_logger

if TYPE_CHECKING:
    from benchmarks.baselines.strategies import Strategy

log = get_logger("cadence.benchmarks.harness")


@dataclass
class BaselineRunConfig:
    window_size: int = 2048
    n_windows: int = 15
    sla_target: float = 0.90
    hardware: HardwareProfile = field(default_factory=HardwareProfile)
    grid: GridProfile = field(default_factory=GridProfile)
    trigger_cfg: DriftTriggerConfig = field(default_factory=DriftTriggerConfig)


@dataclass
class RetrainEvent:
    window_idx: int
    action: str  # "full" | "partial_all_layers" | "no-op"
    pre_f1: float
    post_f1: float
    gpu_seconds: float
    kg_co2: float
    wall_seconds: float


@dataclass
class BaselineOutcome:
    strategy_name: str
    scenario_name: str
    seed: int
    rolling_f1: list[float] = field(default_factory=list)
    events: list[RetrainEvent] = field(default_factory=list)
    total_gpu_seconds: float = 0.0
    total_kg_co2: float = 0.0
    mean_f1: float = 0.0
    min_f1: float = 0.0
    sla_windows_below: int = 0
    sla_recovery_windows: int | None = None  # first window index where F1 >= sla after alert
    forgetting_score: float = 0.0  # baseline_F1 on unrelated segment before vs after
    ground_truth_feature: int = -1
    ground_truth_feature_name: str = ""
    ground_truth_mechanism: str = ""


def make_baseline_stream(
    stream_X: np.ndarray,
    stream_y: np.ndarray,
    scenario: DriftScenario,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, DriftInjector]:
    """Apply the scenario to the raw stream and return (X_drifted, y_drifted, injector)."""
    rng = np.random.default_rng(seed)
    injector = DriftInjector(scenario, rng=rng)
    X_out, y_out = injector.apply(stream_X, stream_y, step_offset=0)
    return X_out, y_out, injector


class Strategy(Protocol):
    name: str

    def decide(
        self,
        *,
        window_idx: int,
        adapter: ModelAdapter,
        window_X: np.ndarray,
        window_y: np.ndarray,
        trigger: PSITrigger,
        rolling_f1: float,
    ) -> str:
        """Return one of {'no-op', 'partial_all_layers', 'full'}."""
        ...


def run_baseline(
    *,
    adapter: ModelAdapter,
    baseline_state: dict,
    baseline_threshold: float,
    historical_X: np.ndarray,
    historical_y: np.ndarray,
    stream_X_raw: np.ndarray,
    stream_y_raw: np.ndarray,
    unrelated_X: np.ndarray,
    unrelated_y: np.ndarray,
    scenario: DriftScenario,
    strategy: Strategy,
    seed: int,
    cfg: BaselineRunConfig,
    feature_names: list[str],
) -> BaselineOutcome:
    """Run one seed of one strategy on one scenario. Returns a `BaselineOutcome`.

    `stream_X_raw`, `stream_y_raw` are the *unseen* production data slice; the
    drift injector is applied to them. `historical_X`, `historical_y` is the
    replay-buffer + full-retrain data (never drifted).
    """
    log.info("baseline_start", strategy=strategy.name, scenario=scenario.name, seed=seed)

    # Reset adapter to the pre-drift baseline for a fair comparison.
    adapter.load_state_dict(baseline_state)
    if hasattr(adapter, "decision_threshold"):
        adapter.decision_threshold = baseline_threshold

    # Baseline F1 on the unrelated segment (for forgetting measurement).
    baseline_unrelated_f1 = _f1_of(adapter, unrelated_X, unrelated_y)

    # Build the drifted stream once — same drift for every strategy so runs are comparable.
    stream_X, stream_y, injector = make_baseline_stream(
        stream_X_raw, stream_y_raw, scenario, seed=seed
    )
    gt = injector.ground_truth()

    trigger = PSITrigger(baseline_X=historical_X, feature_names=feature_names, cfg=cfg.trigger_cfg)

    outcome = BaselineOutcome(
        strategy_name=strategy.name,
        scenario_name=scenario.name,
        seed=seed,
        ground_truth_feature=gt.root_cause_feature_idx,
        ground_truth_feature_name=gt.root_cause_feature_name,
        ground_truth_mechanism=gt.mechanism,
    )

    n_rows = stream_X.shape[0]
    windows_run = 0
    first_recovery = None

    for w in range(cfg.n_windows):
        start = w * cfg.window_size
        end = min(start + cfg.window_size, n_rows)
        if end <= start:
            break
        window_X = stream_X[start:end]
        window_y = stream_y[start:end]
        windows_run += 1

        rolling_f1 = _f1_of(adapter, window_X, window_y)
        outcome.rolling_f1.append(rolling_f1)
        if rolling_f1 < cfg.sla_target:
            outcome.sla_windows_below += 1

        action = strategy.decide(
            window_idx=w,
            adapter=adapter,
            window_X=window_X,
            window_y=window_y,
            trigger=trigger,
            rolling_f1=rolling_f1,
        )

        gpu_s, co2, event_post_f1 = 0.0, 0.0, rolling_f1
        wall_s = 0.0

        if action != "no-op":
            wall_start = time.perf_counter()
            if action == "full":
                cost = estimate_cost(
                    flops_per_forward=adapter.flops_per_forward(),
                    n_samples=historical_X.shape[0] + window_X.shape[0],
                    epochs=15,
                    hardware=cfg.hardware,
                    grid=cfg.grid,
                )
                gpu_s = cost.gpu_seconds
                co2 = cost.kg_co2
                X_full = np.concatenate([historical_X, window_X])
                y_full = np.concatenate([historical_y, window_y])
                adapter.fit(X_full, y_full)
            elif action == "partial_all_layers":  # EWC-only baseline
                cost = estimate_cost(
                    flops_per_forward=adapter.flops_per_forward(),
                    n_samples=window_X.shape[0],
                    epochs=5,
                    hardware=cfg.hardware,
                    grid=cfg.grid,
                )
                gpu_s = cost.gpu_seconds
                co2 = cost.kg_co2
                fisher = getattr(adapter, "compute_fisher", None)
                if fisher is None:
                    raise RuntimeError("EWC-only baseline requires an adapter with compute_fisher")
                fisher_dict = fisher(historical_X[:2000], historical_y[:2000])
                theta_star = {n: p.detach().clone() for n, p in adapter._module.named_parameters()}
                replay_idx = np.random.default_rng(seed + w).integers(
                    0, historical_X.shape[0], size=2000
                )
                adapter.partial_fit(
                    window_X,
                    window_y,
                    layers_to_update=None,  # ALL layers — EWC-only, not causal
                    ewc_penalty=1000.0,
                    fisher=fisher_dict,
                    theta_star=theta_star,
                    replay_X=historical_X[replay_idx],
                    replay_y=historical_y[replay_idx],
                    max_epochs=5,
                )

            wall_s = time.perf_counter() - wall_start
            event_post_f1 = _f1_of(adapter, window_X, window_y)

        outcome.events.append(
            RetrainEvent(
                window_idx=w,
                action=action,
                pre_f1=rolling_f1,
                post_f1=event_post_f1,
                gpu_seconds=gpu_s,
                kg_co2=co2,
                wall_seconds=wall_s,
            )
        )
        outcome.total_gpu_seconds += gpu_s
        outcome.total_kg_co2 += co2

        if first_recovery is None and event_post_f1 >= cfg.sla_target and w > 0:
            first_recovery = w

    outcome.mean_f1 = float(np.mean(outcome.rolling_f1)) if outcome.rolling_f1 else 0.0
    outcome.min_f1 = float(np.min(outcome.rolling_f1)) if outcome.rolling_f1 else 0.0
    outcome.sla_recovery_windows = first_recovery

    # Forgetting on unrelated segment.
    post_unrelated_f1 = _f1_of(adapter, unrelated_X, unrelated_y)
    outcome.forgetting_score = float(baseline_unrelated_f1 - post_unrelated_f1)

    log.info(
        "baseline_done",
        strategy=strategy.name,
        scenario=scenario.name,
        mean_f1=outcome.mean_f1,
        total_gpu_hr=outcome.total_gpu_seconds / 3600.0,
        total_kg_co2=outcome.total_kg_co2,
        forgetting=outcome.forgetting_score,
    )
    return outcome


def _f1_of(adapter: ModelAdapter, X: np.ndarray, y: np.ndarray) -> float:
    probs = adapter.predict_proba(X)
    threshold = getattr(adapter, "decision_threshold", 0.5)
    preds = (probs >= threshold).astype(np.int64)
    return float(f1_score(y, preds, zero_division=0))
