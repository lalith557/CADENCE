"""W-38 guard: PPO must be trained exactly once per seed, NOT once per
(scenario, seed).

The bug (W-38) trained a fresh policy inside the `for scenario` loop, which
(a) multiplied training cost by the number of scenarios (~18 h Stage 1,
~160 h Stage 2) and (b) violated the pre-registered unit of analysis — one
generalist policy per seed, evaluated on all scenarios
(docs/gate_a_preregistration.md Part 2).

This test drives `benchmarks.phase_a_run.main` with all heavy dependencies
mocked and asserts `train_ppo` is called exactly `n_seeds` times regardless of
how many scenarios the run evaluates.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np

import benchmarks.phase_a_run as pa


class _FakeAdapter:
    decision_threshold = 0.5

    def fit(self, *a, **k):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, s):
        return None

    def clone(self):
        return self

    def flops_per_forward(self):
        return 1.0

    def predict_proba(self, X):
        return np.zeros(len(X), dtype="float32")


class _FakeDataset:
    def __init__(self, n=200, d=4):
        rng = np.random.default_rng(0)
        self.X_train = rng.random((n, d)).astype("float32")
        self.y_train = (rng.random(n) > 0.8).astype("int64")
        self.feature_names = [f"f{i}" for i in range(d)]


class _FakeScenario:
    def __init__(self, name):
        self.name = name


class _FakeOutcome:
    name = "stub"
    mean_f1 = 0.5
    min_f1 = 0.4
    total_gpu_seconds = 1.0
    total_kg_co2 = 0.0
    sla_windows_below = 0
    forgetting_score = 0.0
    rolling_f1 = [0.5]
    action_counts = {"no-op": 1, "partial": 0, "full": 0}


_AGG_STUB = {
    "mean_f1": (0.5, 0.0),
    "gpu_hr": (0.0, 0.0),
    "kg_co2": (0.0, 0.0),
    "sla_windows_below": (0.0, 0.0),
}


def _run_main_with_scenarios(n_scenarios: int, n_seeds: int, tmp_path):
    scenarios = [_FakeScenario(f"s{i}") for i in range(n_scenarios)]
    out = tmp_path / "phase_a.json"

    with ExitStack() as es:
        p = lambda name, **kw: es.enter_context(patch.object(pa, name, **kw))  # noqa: E731
        p("load_credit_card_fraud", return_value=_FakeDataset())
        p("FraudNet", return_value=_FakeAdapter())
        p("build_contested_sla_scenarios", return_value=scenarios)
        p("build_step_a_scenarios", return_value=scenarios)
        p("PSITrigger", return_value=MagicMock())
        p("PSIResponsibilityScorer", return_value=MagicMock())
        p("build_multi_scenario_env_factory", return_value=(lambda: MagicMock()))
        mock_train = p("train_ppo", return_value=MagicMock())
        p("run_baseline", return_value=_FakeOutcome())
        p("evaluate_ppo", return_value=_FakeOutcome())
        p(
            "make_baseline_stream",
            return_value=(
                np.zeros((10, 4), dtype="float32"),
                np.zeros(10, dtype="int64"),
                {},
            ),
        )
        p("RetrainingSandboxEnv", return_value=MagicMock())
        p("_aggregate", return_value=_AGG_STUB)
        p("set_global_seed")
        p("mlflow", new=MagicMock())
        mock_start = p("start_run")
        mock_start.return_value.__enter__.return_value = MagicMock()
        mock_start.return_value.__exit__.return_value = False

        rc = pa.main(
            [
                "--config", "configs/default.yaml",
                "--seeds", str(n_seeds),
                "--per-seed-policies",
                "--contested-only",
                "--train-timesteps", "10",
                "--n-windows", "2",
                "--window-size", "8",
                "--out", str(out),
            ]
        )
    return rc, mock_train


def test_train_ppo_called_once_per_seed_regardless_of_scenarios(tmp_path):
    # 3 scenarios, 2 seeds. The W-38 bug would call train_ppo 2*3 = 6 times.
    rc, mock_train = _run_main_with_scenarios(n_scenarios=3, n_seeds=2, tmp_path=tmp_path)
    assert rc == 0
    assert mock_train.call_count == 2, (
        f"train_ppo called {mock_train.call_count}x for 2 seeds x 3 scenarios; "
        "must be 2 (once per seed). W-38 regression."
    )


def test_train_ppo_scales_with_seeds_not_scenarios(tmp_path):
    # 5 scenarios, 3 seeds -> must be 3 (per seed), never 15.
    _, mock_train = _run_main_with_scenarios(n_scenarios=5, n_seeds=3, tmp_path=tmp_path)
    assert mock_train.call_count == 3
