"""Remaining §4 fallbacks (§4.4, §4.6, §4.7, §4.8).

Each fallback is a pure function or small stateful helper that inspects a
proposed action / responsibility scores / telemetry and returns either
(a) the same action unchanged, or (b) a *safer* fallback action. Composed
above `RetrainExecutor.execute(...)` by `EscalationLadder` and by the RSO
driver in `benchmarks.phase_c_run`.

Design notes:
  * These functions never raise — they log and adjust. The only exception
    is `resource_fallback` which catches CUDA OOMs from a callable.
  * Each has a matching test in `tests/unit/test_fallbacks.py` that forces
    the failure and asserts the fallback fires.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Callable

import numpy as np

from cadence.common.logging import get_logger

log = get_logger("cadence.executor.fallbacks")


# ---------- §4.4 diffuse-responsibility fallback ----------


def herfindahl_concentration(scores: np.ndarray) -> float:
    """Same normalized Herfindahl as `cadence.rso.env._herfindahl` — kept in
    sync deliberately so the RSO state feature and this fallback both see
    the same "how concentrated is the attribution" signal.
    """
    s = np.asarray(scores, dtype=np.float64)
    total = s.sum()
    if total <= 0 or s.size == 0:
        return 0.0
    p = s / total
    h = float((p * p).sum())
    n = float(s.size)
    if n <= 1:
        return 1.0
    return max(0.0, min(1.0, (h - 1.0 / n) / (1.0 - 1.0 / n)))


@dataclass
class DiffuseFallbackConfig:
    tau_concentration: float = 0.15  # < tau => "no single node dominates"


def diffuse_responsibility_fallback(
    action: str,
    scores: np.ndarray,
    *,
    cfg: DiffuseFallbackConfig | None = None,
) -> str:
    """§4.4: when responsibility is spread across many nodes (Herfindahl
    concentration < τ), a partial retrain of a single layer isn't the right
    intervention — escalate to a full retrain instead.

    Returns the (possibly-fallback) action name.
    """
    cfg = cfg or DiffuseFallbackConfig()
    if action != "partial":
        return action
    concentration = herfindahl_concentration(scores)
    if concentration < cfg.tau_concentration:
        log.warning(
            "diffuse_responsibility_fallback",
            concentration=concentration,
            threshold=cfg.tau_concentration,
        )
        return "full"
    return action


# ---------- §4.6 trigger false-alarm handling ----------


@dataclass
class FalseAlarmConfig:
    sla_target: float = 0.65


def is_false_alarm(
    psi_alert_fired: bool,
    delayed_label_f1: float | None,
    *,
    cfg: FalseAlarmConfig | None = None,
) -> bool:
    """§4.6: PSI alert fires but the delayed-label F1 is above SLA -> no-op.

    Returns True if the alert is a false positive and the caller should
    ignore it. `delayed_label_f1` is the ground-truth F1 measured on a
    delayed-label slice; None means labels aren't back yet (treat as alert).
    """
    cfg = cfg or FalseAlarmConfig()
    if not psi_alert_fired:
        return False
    if delayed_label_f1 is None:
        return False
    if delayed_label_f1 >= cfg.sla_target:
        log.info(
            "psi_false_alarm_suppressed",
            delayed_label_f1=delayed_label_f1,
            sla_target=cfg.sla_target,
        )
        return True
    return False


# ---------- §4.7 resource fallback (OOM / timeout) ----------


class ResourceFallbackError(RuntimeError):
    """Raised when the resource fallback fires — a signal to the RSO that
    this retrain didn't run, so the old model keeps serving.
    """


def resource_fallback(fn: Callable[[], object]) -> object:
    """§4.7: wrap a callable; catch CUDA OOMs and time-outs cleanly.

    On OOM:
      * gc.collect + torch.cuda.empty_cache
      * log a warning
      * raise ResourceFallbackError so the caller knows to keep the old
        model in service (§4.7: never crash the loop, never silently serve
        a partially-updated model).

    Non-OOM exceptions propagate as-is.
    """
    try:
        import torch  # noqa: PLC0415 — local import so the module has no hard torch dep
    except Exception:  # noqa: BLE001
        return fn()

    try:
        return fn()
    except torch.cuda.OutOfMemoryError as e:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.error("resource_fallback_oom", err=str(e))
        raise ResourceFallbackError(f"CUDA OOM aborted retrain: {e}") from e
    except TimeoutError as e:
        log.error("resource_fallback_timeout", err=str(e))
        raise ResourceFallbackError(f"timeout aborted retrain: {e}") from e


# ---------- §4.8 cold-start / missing telemetry ----------


def cold_start_scores(
    activation_traces: dict | None,
    feature_scores: np.ndarray | None,
    *,
    n_features: int,
) -> np.ndarray:
    """§4.8: when activation traces are unavailable (cold-start on a fresh
    deployment, or the telemetry pipeline dropped them), degrade to a
    feature-only attribution vector instead of hard-failing.

    Inputs:
      * `activation_traces`: dict[layer_name -> array]. None means missing.
      * `feature_scores`: per-feature score vector, length `n_features`.
        None means the feature-side telemetry is also missing.

    Returns a per-feature score in [0, 1] of length `n_features`.
    """
    if activation_traces is not None and len(activation_traces) > 0 and feature_scores is not None:
        # Full telemetry — caller is expected to run the normal scorer;
        # this function returns feature_scores unchanged as a passthrough.
        return np.asarray(feature_scores, dtype=np.float32)

    if feature_scores is not None:
        log.warning(
            "cold_start_feature_only_attribution",
            reason="activation_traces missing; using feature-only scores",
        )
        return np.asarray(feature_scores, dtype=np.float32)

    log.warning(
        "cold_start_uniform_attribution",
        reason="both activation and feature telemetry missing; uniform priors",
    )
    return np.full(n_features, 1.0 / max(n_features, 1), dtype=np.float32)
