from pathlib import Path

import mlflow

from cadence.common.config import Config
from cadence.common.tracking import start_run


def test_start_run_logs_config_hash(tmp_path: Path):
    cfg = Config(profile="ci")
    cfg.experiment.mlflow_uri = f"file:{tmp_path / 'mlruns'}"
    cfg.experiment.experiment_name = "test-exp"

    with start_run(cfg, run_name="unit-test") as run:
        run_id = run.info.run_id

    mlflow.set_tracking_uri(cfg.experiment.mlflow_uri)
    finished = mlflow.get_run(run_id)
    assert finished.data.tags.get("cadence.profile") == "ci"
    assert finished.data.tags.get("cadence.config_hash") == cfg.hash()
    assert finished.data.params.get("seed") == "42"
