"""MLflow tracking wrapper.

Central rule: every experiment we care about starts via `start_run(cfg)`, which
logs the config's hash + git SHA + seed as tags/params so runs are reproducible
from just their run_id.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# mlflow >=3 refuses filesystem tracking backends by default. Our default
# tracking URI is `file:./experiments/mlruns` (per configs/*.yaml) — keep
# that lightweight local-file workflow by opting in before mlflow is imported.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow  # noqa: E402  (needs the env var set first)

from cadence.common.config import Config  # noqa: E402


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parents[2],
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except FileNotFoundError:
        pass
    return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parents[2],
        )
        return bool(out.stdout.strip())
    except FileNotFoundError:
        return False


@contextmanager
def start_run(cfg: Config, run_name: str | None = None) -> Iterator[mlflow.ActiveRun]:
    """Open an MLflow run with the canonical set of tags/params logged."""
    mlflow.set_tracking_uri(cfg.experiment.mlflow_uri)
    mlflow.set_experiment(cfg.experiment.experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("cadence.profile", cfg.profile)
        mlflow.set_tag("cadence.config_hash", cfg.hash())
        mlflow.set_tag("cadence.git_sha", _git_sha())
        mlflow.set_tag("cadence.git_dirty", str(_git_dirty()).lower())
        mlflow.log_param("seed", cfg.experiment.seed)
        mlflow.log_dict(cfg.model_dump(mode="json"), "config.json")
        yield run
