"""Resumability tests for run_experiment.py.

Two properties must hold, and both are pinned here:

1. **Atomic ledger writes.** A kill mid-write cannot corrupt the ledger — we
   write to a temp file in the same directory, then `os.replace` in.

2. **Resume skips COMPLETED seeds and picks up the INTERRUPTED one.** After a
   simulated kill on seed 1 (with seed 0 already COMPLETED and seed 2
   NOT_STARTED), the next invocation must attempt seed 1 and then seed 2 — not
   seed 0.

We stub `run_one_seed` so no real PPO training happens; the resumability logic
under test does not depend on the training payload.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Import the module under test. run_experiment.py lives at the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import run_experiment  # noqa: E402


def _fresh_ledger_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    ledger = tmp_path / "gate_a_ledger.json"
    monkeypatch.setattr(run_experiment, "LEDGER_PATH", ledger)
    return ledger


def test_atomic_write_json_round_trips_and_leaves_no_temp(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "state.json"
    payload = {"a": 1, "nested": {"b": [1, 2, 3]}}
    run_experiment.atomic_write_json(path, payload)
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    # No stray *.tmp files left behind by the atomic writer.
    stray = list(path.parent.glob(f"{path.name}.*.tmp"))
    assert stray == [], f"atomic_write_json left temp files: {stray}"


def test_resume_skips_completed_and_picks_up_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Simulate a killed-mid-stage state and verify --resume behaviour."""
    ledger_path = _fresh_ledger_path(tmp_path, monkeypatch)

    # Seed the ledger as if a prior session completed seed 0, was interrupted
    # on seed 1, and never got to seed 2.
    ledger = {
        "experiment_id": "gate-a-testfake-testfake",
        "git_commit": "testfake",
        "config_hash": "testfake",
        "config_path": "configs/gate_a.yaml",
        "created": "2026-09-01T00:00:00+00:00",
        "stages": {
            "1": {
                "params": {"seeds": 3, "train_timesteps": 100, "n_windows": 2,
                           "window_size": 64, "sla": 0.65, "contested_only": True},
                "started": "2026-09-01T00:00:00+00:00",
                "seeds": [
                    {"seed": 0, "status": "COMPLETED",
                     "result_path": "dummy", "wall_seconds": 1.0},
                    {"seed": 1, "status": "INTERRUPTED",
                     "started": "2026-09-01T00:01:00+00:00"},
                    {"seed": 2, "status": "NOT_STARTED"},
                ],
            }
        },
    }
    run_experiment.atomic_write_json(ledger_path, ledger)

    # Stub run_one_seed so no real training happens. Record which seeds get called.
    called: list[int] = []

    def fake_run(config_path, stage, seed, params):
        called.append(seed)
        # Write a plausible result JSON so extract_metrics doesn't blow up.
        out_dir = tmp_path / f"gate_a_stage{stage}"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"seed_{seed}.json"
        out_path.write_text(
            json.dumps({"scenarios": {"scen_a": {"aggregates": {
                "rso_ppo": {"mean_f1": [0.7, 0.01], "gpu_hr": [0.001, 0.0001]}
            }}}})
        )
        return 0, out_path, 0.1

    # Reload ledger from disk (mimics real invocation) + run resume.
    ledger = run_experiment.load_ledger()
    assert ledger is not None
    run_experiment.run_stage(
        ledger,
        stage=1,
        config_path=Path("configs/gate_a.yaml"),
        dry_run=False,
        only_seed=None,
        run_one_seed=fake_run,
    )

    # Must have attempted seeds 1 and 2 (in order), not seed 0.
    assert called == [1, 2], f"resume order wrong: {called}"

    # Ledger on disk must reflect the new COMPLETED statuses atomically.
    final_ledger = run_experiment.load_ledger()
    assert final_ledger is not None
    seed_states = {s["seed"]: s["status"] for s in final_ledger["stages"]["1"]["seeds"]}
    assert seed_states == {0: "COMPLETED", 1: "COMPLETED", 2: "COMPLETED"}, seed_states

    # Metrics were extracted for the seeds we ran.
    for entry in final_ledger["stages"]["1"]["seeds"]:
        if entry["seed"] in (1, 2):
            assert "metrics" in entry, f"seed {entry['seed']} missing metrics"


def test_interrupted_seed_marks_ledger_and_exits_130(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A KeyboardInterrupt inside run_one_seed marks the seed INTERRUPTED (never PASS/FAIL)."""
    ledger_path = _fresh_ledger_path(tmp_path, monkeypatch)
    ledger = run_experiment.init_ledger(Path("configs/gate_a.yaml"), "abc12345", "cfg1")
    run_experiment.atomic_write_json(ledger_path, ledger)

    def raise_sigint(config_path, stage, seed, params):
        raise KeyboardInterrupt

    ledger = run_experiment.load_ledger()
    assert ledger is not None
    with pytest.raises(SystemExit) as excinfo:
        run_experiment.run_stage(
            ledger,
            stage=1,
            config_path=Path("configs/gate_a.yaml"),
            dry_run=False,
            only_seed=None,
            run_one_seed=raise_sigint,
        )
    assert excinfo.value.code == 130

    final = run_experiment.load_ledger()
    assert final is not None
    first_seed = final["stages"]["1"]["seeds"][0]
    assert first_seed["status"] == "INTERRUPTED", first_seed
