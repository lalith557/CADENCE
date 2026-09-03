"""W-40: RewardComponentCallback logs the metrics gate evaluator needs.

Pre-registered gates G1, G3, G4, G5 require training-time visibility into:
  * action distribution (G1: max_pct <= 0.85 on last 30% -> no action collapse)
  * reward-component decomposition (G3: reward improves; G4: cost visible;
    G5: not SLA-gaming — cost + SLA penalties both shrink)

Before W-40 the runner's only MLflow surface was SB3's own logger via
MLflowCallback (ep_rew_mean, entropy_loss, ...). None of the four above
were queryable, forcing every gate to be TBD post-run — the exact
'blind spot' R-Gate-A-stage1-fail-1 documented.

This test synthesises 20 fake env-info dicts (10 partial + 10 no-op)
via the callback's `_on_step` path and asserts the aggregate emitted at
flush time hits the metric names the gate evaluator queries. We stub
`mlflow.log_metric` so the test does not require an MLflow tracking URI.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


def _make_callback_at_num_timesteps(monkeypatch, num_timesteps: int):
    """Instantiate RewardComponentCallback and set num_timesteps to trigger flush."""
    from cadence.rso.ppo import RewardComponentCallback

    cb = RewardComponentCallback(log_every=1)
    # SB3's BaseCallback exposes num_timesteps as a property backed by self.model.
    # Monkeypatch the internal attribute — cheaper than standing up a fake model.
    cb.__dict__["num_timesteps"] = num_timesteps
    # locals is set by SB3 during learn(); we set it manually.
    cb.locals = {}
    return cb


def test_reward_component_callback_flushes_expected_metric_names(monkeypatch) -> None:
    from cadence.rso import ppo as ppo_mod

    cb = _make_callback_at_num_timesteps(monkeypatch, num_timesteps=1)

    # Simulate 20 env steps across two info-dict shapes.
    for i in range(20):
        action = 1 if i < 10 else 0
        cb.locals["infos"] = [
            {
                "delta_f1_component": 0.05 if action else 0.0,
                "cost_penalty_component": 5e-4 if action else 0.0,
                "sla_penalty_component": 0.02,
                "action": action,
                "lambda_sla": 3.0,
            }
        ]
        # Bypass the "log_every" throttle by making each _on_step at same step
        # accumulate first, then flush on the last call by bumping num_timesteps.
        cb.__dict__["num_timesteps"] = 0 if i < 19 else 1
        with patch.object(ppo_mod.mlflow, "log_metric") as mock_log:
            cb._on_step()
        if i < 19:
            # Not flushed yet.
            assert not mock_log.called, f"flushed prematurely at i={i}"

    # After the 20th step (with num_timesteps bumped), flush fires.
    expected_names = {
        "reward/delta_f1_avg",
        "reward/cost_penalty_avg",
        "reward/sla_penalty_avg",
        "reward/cost_visible_ratio",
        "reward/lambda_sla_avg",
        "action/no_op_pct",
        "action/partial_pct",
        "action/full_pct",
        "action/max_pct",
    }
    logged_names = {call.args[0] for call in mock_log.call_args_list}
    missing = expected_names - logged_names
    extra_action = {n for n in logged_names if n.startswith("action/")} - expected_names
    assert not missing, f"callback missed metric names: {sorted(missing)}"
    # Verify the numerical semantics gate evaluator relies on:
    logged_map = {call.args[0]: call.args[1] for call in mock_log.call_args_list}
    # 10 partial + 10 no-op -> max_pct should be 0.5 (tie between the two, either counts)
    assert 0.45 <= logged_map["action/max_pct"] <= 0.55, logged_map["action/max_pct"]
    assert 0.45 <= logged_map["action/no_op_pct"] <= 0.55, logged_map["action/no_op_pct"]
    # cost_visible_ratio should be nonzero (cost_pen > 0 on the 10 partial steps).
    assert logged_map["reward/cost_visible_ratio"] > 0.0
    # lambda_sla_avg should equal 3.0 (constant across all 20 infos).
    assert 2.9 < logged_map["reward/lambda_sla_avg"] < 3.1


def test_reward_component_callback_survives_missing_info_keys(monkeypatch) -> None:
    """Older env versions without info['delta_f1_component'] must not crash."""
    from cadence.rso import ppo as ppo_mod

    cb = _make_callback_at_num_timesteps(monkeypatch, num_timesteps=1)
    cb.locals["infos"] = [{"pre_f1": 0.5, "post_f1": 0.6, "cost_gpu_hr": 0.0}]
    with patch.object(ppo_mod.mlflow, "log_metric") as mock_log:
        cb._on_step()  # Should not raise.
    # No delta_f1_component => the accumulator stays at 0; the flush still
    # fires and logs zero-valued metrics (n=1 fallback), but must NOT crash.
    assert mock_log.called  # flush happened
