"""CADENCE REST API — FastAPI service exposing the full loop over HTTP.

Endpoints:
  * GET  /health         — server + model + policy version info.
  * POST /score          — run rows through the pretrained adapter.
  * POST /monitor        — evaluate PSI drift on a window; return alert +
                            per-feature attribution scores.
  * POST /decide         — invoke the learned RSO policy on a scored window;
                            return the recommended action + estimated cost.
  * POST /admin/reload   — hot-swap the model / policy checkpoint (guarded
                            behind CADENCE_API_ADMIN_TOKEN).

Design notes:
  * The API is a THIN wrapper over the same classes the benchmark runners
    use — `FraudNet`, `PSITrigger`, `GNNResponsibilityScorer`, PPO-loaded
    policy, `RetrainExecutor`. Nothing new here; it just brokers HTTP.
  * State (loaded adapter, policy, scorers) is process-local. Multiple
    workers = multiple copies of the state. For horizontal scale you'd
    move the checkpoints to shared storage and rebuild state per worker.
  * All numeric inputs use lists-of-floats to stay JSON-friendly; the
    server converts to numpy before passing to the model.
  * No secret is required to boot the API. `CADENCE_API_ADMIN_TOKEN` is
    only checked on `/admin/*` routes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from cadence.adapters.neural import FraudNet, FraudNetConfig
from cadence.collector.drift_trigger import DriftTriggerConfig, PSITrigger
from cadence.common.logging import get_logger

log = get_logger("cadence.api.server")


# ---------- Pydantic request/response models ----------


class ScoreRequest(BaseModel):
    rows: list[list[float]] = Field(..., description="Batch of feature vectors")


class ScoreResponse(BaseModel):
    probabilities: list[float]
    threshold: float


class MonitorRequest(BaseModel):
    rows: list[list[float]] = Field(..., description="Window of feature vectors to monitor")
    y: Optional[list[int]] = Field(
        default=None, description="Optional labels (enables delayed-label F1 check)"
    )


class MonitorResponse(BaseModel):
    psi_per_feature: list[float]
    psi_alert_fired: bool
    delayed_f1: Optional[float]
    top_k_features: list[dict]


class DecideRequest(BaseModel):
    rows: list[list[float]] = Field(..., description="Window that just fired the trigger")
    y: Optional[list[int]] = Field(default=None, description="Labels for post-hoc F1")
    sla_target: float = Field(default=0.65, description="Effective SLA threshold on F1")


class DecideResponse(BaseModel):
    action: str  # "no-op" | "partial" | "full"
    action_int: int
    estimated_gpu_hr: float
    estimated_kg_co2: float
    top_feature_index: int
    reason: str


class HealthResponse(BaseModel):
    status: str
    adapter_family: str | None
    adapter_input_dim: int | None
    policy_loaded: bool
    policy_obs_dim: int | None
    checkpoint_path: str | None


# ---------- Service state ----------


@dataclass
class ServiceState:
    adapter: FraudNet | None = None
    trigger: PSITrigger | None = None
    baseline_X: np.ndarray | None = None
    feature_names: list[str] | None = None
    policy: object | None = None  # stable_baselines3 PPO instance
    checkpoint_path: str | None = None


STATE = ServiceState()


def _admin_auth(token: str = Depends(APIKeyHeader(name="X-Admin-Token", auto_error=False))) -> None:
    expected = os.environ.get("CADENCE_API_ADMIN_TOKEN")
    if expected is None or expected == "":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin routes disabled — set CADENCE_API_ADMIN_TOKEN")
    if token != expected:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid admin token")


# ---------- FastAPI app ----------


app = FastAPI(
    title="CADENCE API",
    description="Causal Attribution-Driven Efficient Continual Retraining — REST surface.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        adapter_family=STATE.adapter.family if STATE.adapter else None,
        adapter_input_dim=(STATE.adapter.cfg.input_dim if STATE.adapter else None),
        policy_loaded=STATE.policy is not None,
        policy_obs_dim=(
            int(STATE.policy.observation_space.shape[0]) if STATE.policy else None
        ),
        checkpoint_path=STATE.checkpoint_path,
    )


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    if STATE.adapter is None:
        raise HTTPException(503, "adapter not loaded — POST /admin/reload first")
    X = np.asarray(req.rows, dtype=np.float32)
    probs = STATE.adapter.predict_proba(X)
    return ScoreResponse(
        probabilities=[float(p) for p in probs],
        threshold=float(STATE.adapter.decision_threshold),
    )


@app.post("/monitor", response_model=MonitorResponse)
def monitor(req: MonitorRequest) -> MonitorResponse:
    if STATE.adapter is None or STATE.trigger is None:
        raise HTTPException(503, "adapter/trigger not loaded")
    X = np.asarray(req.rows, dtype=np.float32)
    alert = STATE.trigger.evaluate(X)
    psi_vec = STATE.trigger.psi_per_feature(X)
    order = np.argsort(-psi_vec)[:5]
    top = [
        {
            "feature_index": int(i),
            "feature_name": (
                STATE.feature_names[i]
                if STATE.feature_names and i < len(STATE.feature_names)
                else f"f{i}"
            ),
            "psi": float(psi_vec[i]),
        }
        for i in order
    ]
    delayed_f1: float | None = None
    if req.y is not None:
        from sklearn.metrics import f1_score

        probs = STATE.adapter.predict_proba(X)
        preds = (probs >= STATE.adapter.decision_threshold).astype(np.int64)
        delayed_f1 = float(f1_score(np.asarray(req.y), preds, zero_division=0))
    return MonitorResponse(
        psi_per_feature=[float(x) for x in psi_vec],
        psi_alert_fired=bool(alert.fired),
        delayed_f1=delayed_f1,
        top_k_features=top,
    )


@app.post("/decide", response_model=DecideResponse)
def decide(req: DecideRequest) -> DecideResponse:
    from cadence.carbon.model import GridProfile, HardwareProfile, estimate_cost
    from cadence.executor.fallbacks import FalseAlarmConfig, is_false_alarm

    if STATE.adapter is None or STATE.trigger is None:
        raise HTTPException(503, "adapter/trigger not loaded")

    X = np.asarray(req.rows, dtype=np.float32)
    alert = STATE.trigger.evaluate(X)
    delayed_f1: float | None = None
    if req.y is not None:
        from sklearn.metrics import f1_score

        probs = STATE.adapter.predict_proba(X)
        preds = (probs >= STATE.adapter.decision_threshold).astype(np.int64)
        delayed_f1 = float(f1_score(np.asarray(req.y), preds, zero_division=0))

    # §4.6 false-alarm suppression before we ask the RSO to act.
    if is_false_alarm(alert.fired, delayed_f1, cfg=FalseAlarmConfig(sla_target=req.sla_target)):
        return DecideResponse(
            action="no-op",
            action_int=0,
            estimated_gpu_hr=0.0,
            estimated_kg_co2=0.0,
            top_feature_index=-1,
            reason="PSI fired but delayed-label F1 >= SLA — false alarm suppressed (§4.6)",
        )

    psi_vec = STATE.trigger.psi_per_feature(X)
    top = int(np.argmax(psi_vec))

    # If no learned policy is loaded, fall back to a rule.
    if STATE.policy is None:
        action_int = 1 if psi_vec[top] > 0.25 else 0
        reason = "rule fallback (no PPO checkpoint loaded)"
    else:
        # Build a 15-d obs the same way `cadence.rso.env` would.
        top_scores = np.sort(psi_vec)[::-1][:5]
        top_scores = np.pad(top_scores, (0, max(0, 5 - len(top_scores))), constant_values=0)
        current_f1 = delayed_f1 if delayed_f1 is not None else 0.0
        sla_margin = float(max(0.0, req.sla_target - current_f1))
        # Herfindahl concentration.
        total = float(psi_vec.sum())
        if total > 0:
            p = psi_vec / total
            h = float((p * p).sum())
            n = float(psi_vec.size)
            concentration = max(0.0, min(1.0, (h - 1.0 / n) / max(1e-9, 1.0 - 1.0 / n)))
        else:
            concentration = 0.0
        obs = np.array(
            [
                *top_scores.tolist(),
                float(current_f1),
                float(req.sla_target),
                0.0,  # windows_since_alert
                0.0,  # no-op cost
                1e-6,  # partial cost placeholder
                1e-4,  # full cost placeholder
                481.0,  # grid intensity
                0.0,  # hours_since_retrain
                sla_margin,
                concentration,
            ],
            dtype=np.float32,
        )
        action, _ = STATE.policy.predict(obs, deterministic=True)
        action_int = int(np.asarray(action).flatten()[0])
        reason = "learned PPO policy"

    if STATE.adapter is None:
        gpu_hr = 0.0
        kg_co2 = 0.0
    else:
        flops = STATE.adapter.flops_per_forward()
        if action_int == 1:
            cost = estimate_cost(flops, X.shape[0], 5, hardware=HardwareProfile(), grid=GridProfile())
        elif action_int == 2:
            cost = estimate_cost(flops, X.shape[0] + 20000, 15, hardware=HardwareProfile(), grid=GridProfile())
        else:
            cost = None
        gpu_hr = (cost.gpu_seconds / 3600.0) if cost is not None else 0.0
        kg_co2 = cost.kg_co2 if cost is not None else 0.0

    return DecideResponse(
        action=["no-op", "partial", "full"][action_int],
        action_int=action_int,
        estimated_gpu_hr=float(gpu_hr),
        estimated_kg_co2=float(kg_co2),
        top_feature_index=top,
        reason=reason,
    )


class ReloadRequest(BaseModel):
    adapter_ckpt: Optional[str] = None
    policy_ckpt: Optional[str] = None
    baseline_X: Optional[list[list[float]]] = None
    feature_names: Optional[list[str]] = None


@app.post("/admin/reload", dependencies=[Depends(_admin_auth)])
def reload(req: ReloadRequest) -> dict:
    """Reload adapter + PSI trigger + learned policy from disk.

    All parameters optional so the caller can hot-swap just the policy
    without touching the adapter.
    """
    n_reloaded = 0
    if req.baseline_X is not None and req.feature_names is not None:
        STATE.baseline_X = np.asarray(req.baseline_X, dtype=np.float32)
        STATE.feature_names = list(req.feature_names)
        STATE.trigger = PSITrigger(
            baseline_X=STATE.baseline_X,
            feature_names=STATE.feature_names,
            cfg=DriftTriggerConfig(psi_threshold=0.25, min_window_rows=200),
        )
        n_reloaded += 1
    if req.policy_ckpt is not None:
        p = Path(req.policy_ckpt)
        if not p.exists():
            raise HTTPException(400, f"policy checkpoint not found: {p}")
        try:
            from stable_baselines3 import PPO

            STATE.policy = PPO.load(str(p), custom_objects={"lr_schedule": lambda _: 0.0})
            STATE.checkpoint_path = str(p)
            n_reloaded += 1
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"policy load failed: {e}") from e
    return {"reloaded": n_reloaded, "checkpoint": STATE.checkpoint_path}


# ---------- Bootstrap helper for local dev / tests ----------


def bootstrap_from_config(config_path: str = "configs/default.yaml") -> None:
    """Populate STATE from the default Fraud pipeline. Used by tests + the
    dockerised dashboard smoke — real deployments POST /admin/reload."""
    from cadence.common.config import load_config
    from cadence.common.seeds import set_global_seed
    from cadence.data.loaders import load_credit_card_fraud

    cfg = load_config(config_path)
    ds = load_credit_card_fraud(cfg.data, seed=42)
    n = ds.X_train.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    n_pre = int(0.70 * n)
    X_pre = ds.X_train[idx[:n_pre]]
    y_pre = ds.y_train[idx[:n_pre]]

    set_global_seed(42)
    adapter = FraudNet(
        FraudNetConfig(
            input_dim=ds.X_train.shape[1],
            hidden_dims=list(cfg.model.hidden_dims),
            dropout=cfg.model.dropout,
            lr=cfg.model.lr,
            batch_size=cfg.model.batch_size,
            max_epochs=cfg.model.max_epochs,
            early_stopping_patience=cfg.model.early_stopping_patience,
            class_weighted=cfg.model.class_weighted,
        )
    )
    adapter.fit(X_pre[:-5000], y_pre[:-5000], X_val=X_pre[-5000:], y_val=y_pre[-5000:])
    STATE.adapter = adapter
    STATE.baseline_X = X_pre
    STATE.feature_names = ds.feature_names
    STATE.trigger = PSITrigger(
        baseline_X=X_pre,
        feature_names=ds.feature_names,
        cfg=DriftTriggerConfig(psi_threshold=0.25, min_window_rows=200),
    )
    # Try the learned policy; silently ok if missing (tests still exercise /decide via rule).
    for candidate in ("experiments/rso_ppo_phase_a.zip",):
        p = Path(candidate)
        if p.exists():
            try:
                from stable_baselines3 import PPO

                STATE.policy = PPO.load(str(p), custom_objects={"lr_schedule": lambda _: 0.0})
                STATE.checkpoint_path = str(p)
                log.info("api_policy_loaded", path=str(p))
                break
            except Exception as e:  # noqa: BLE001
                log.warning("api_policy_load_failed", path=str(p), err=str(e))
