"""REST API integration tests (P2)."""

from __future__ import annotations

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
starlette = pytest.importorskip("starlette")

from fastapi.testclient import TestClient  # noqa: E402

from cadence.api.server import STATE, app  # noqa: E402
from cadence.collector.drift_trigger import DriftTriggerConfig, PSITrigger  # noqa: E402


class _StubAdapter:
    """Minimal adapter shim so we don't need to train FraudNet for API tests."""

    family = "stub"

    class _Cfg:
        input_dim = 4

    cfg = _Cfg()
    decision_threshold = 0.5

    def predict_proba(self, X):
        # Deterministic mapping: probability = 1 - exp(-|X[:, 0]|). No fit required.
        X = np.asarray(X, dtype=np.float32)
        return (1.0 - np.exp(-np.abs(X[:, 0]))).astype(np.float32)

    def flops_per_forward(self) -> int:
        return 100


def _prep_state() -> None:
    STATE.adapter = _StubAdapter()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((512, 4)).astype(np.float32)
    STATE.baseline_X = X
    STATE.feature_names = ["a", "b", "c", "d"]
    STATE.trigger = PSITrigger(
        baseline_X=X,
        feature_names=STATE.feature_names,
        cfg=DriftTriggerConfig(psi_threshold=0.25, min_window_rows=64),
    )
    STATE.policy = None
    STATE.checkpoint_path = None


@pytest.fixture
def client() -> TestClient:
    _prep_state()
    return TestClient(app)


def test_health_reports_loaded_state(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["adapter_family"] == "stub"
    assert body["adapter_input_dim"] == 4
    assert body["policy_loaded"] is False


def test_score_returns_probabilities(client: TestClient) -> None:
    rows = [[1.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0]]
    r = client.post("/score", json={"rows": rows})
    assert r.status_code == 200
    body = r.json()
    assert len(body["probabilities"]) == 2
    assert 0.0 <= body["probabilities"][0] <= 1.0


def test_monitor_returns_psi_and_top_k(client: TestClient) -> None:
    rng = np.random.default_rng(7)
    # Introduce a drift on feature 2.
    live = rng.standard_normal((300, 4)).astype(np.float32)
    live[:, 2] = live[:, 2] * 3.0 + 5.0
    r = client.post("/monitor", json={"rows": live.tolist()})
    assert r.status_code == 200
    body = r.json()
    assert len(body["psi_per_feature"]) == 4
    assert body["top_k_features"][0]["feature_index"] in {2}
    assert isinstance(body["psi_alert_fired"], bool)


def test_decide_returns_action(client: TestClient) -> None:
    rng = np.random.default_rng(1)
    live = rng.standard_normal((256, 4)).astype(np.float32)
    live[:, 0] = live[:, 0] * 3.0 + 4.0
    y = (rng.random(256) < 0.5).astype(int).tolist()
    r = client.post("/decide", json={"rows": live.tolist(), "y": y, "sla_target": 0.70})
    assert r.status_code == 200
    body = r.json()
    assert body["action"] in {"no-op", "partial", "full"}
    assert 0 <= body["action_int"] <= 2
    assert body["estimated_gpu_hr"] >= 0.0
    assert body["top_feature_index"] >= 0


def test_admin_reload_requires_token(client: TestClient) -> None:
    r = client.post("/admin/reload", json={})
    assert r.status_code in {401, 403}  # no token env var set in tests


def test_score_fails_when_adapter_unloaded(client: TestClient) -> None:
    STATE.adapter = None
    r = client.post("/score", json={"rows": [[0.0, 0.0, 0.0, 0.0]]})
    assert r.status_code == 503
