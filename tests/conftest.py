from __future__ import annotations

from pathlib import Path

import pytest

from cadence.common.config import Config, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def ci_config() -> Config:
    return load_config(PROJECT_ROOT / "configs" / "ci.yaml")


@pytest.fixture
def default_config() -> Config:
    return load_config(PROJECT_ROOT / "configs" / "default.yaml")
