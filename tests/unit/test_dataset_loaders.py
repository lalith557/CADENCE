"""Loader smoke tests. Each loader is marked `needs_dataset` so CI can skip when
the raw file isn't present."""

from pathlib import Path

import pytest

from cadence.common.config import Config
from cadence.data.loaders import (
    DatasetNotFoundError,
    load_adult,
    load_credit_card_fraud,
    load_give_me_some_credit,
    load_telco_churn,
)


@pytest.mark.needs_dataset
def test_load_credit_card_fraud(default_config: Config):
    if not default_config.data.fraud_csv.exists():
        pytest.skip("creditcard.csv not present")
    ds = load_credit_card_fraud(default_config.data, seed=default_config.experiment.seed)
    assert ds.X_train.shape[1] == 30  # Time + Amount + V1..V28
    assert ds.name == "credit_card_fraud"
    assert 0.0 < ds.summary()["positive_rate_train"] < 0.01  # ~0.17% fraud


@pytest.mark.needs_dataset
def test_load_give_me_some_credit(default_config: Config):
    if not default_config.data.gmsc_train_csv.exists():
        pytest.skip("cs-training.csv not present")
    ds = load_give_me_some_credit(default_config.data, seed=default_config.experiment.seed)
    assert ds.X_train.shape[1] == 10
    assert ds.name == "give_me_some_credit"


@pytest.mark.needs_dataset
def test_load_telco_churn(default_config: Config):
    if not default_config.data.telco_csv.exists():
        pytest.skip("telco csv not present")
    ds = load_telco_churn(default_config.data, seed=default_config.experiment.seed)
    assert ds.X_train.shape[0] > 0
    assert ds.name == "telco_churn"


@pytest.mark.needs_dataset
def test_load_adult(default_config: Config):
    if not (default_config.data.adult_dir / "adult.data").exists():
        pytest.skip("adult.data not present")
    ds = load_adult(default_config.data, seed=default_config.experiment.seed)
    assert ds.X_train.shape[0] > 0
    assert ds.name == "adult"


def test_missing_dataset_raises_with_url(tmp_path: Path):
    cfg = Config()
    cfg.data.fraud_csv = tmp_path / "does_not_exist.csv"
    with pytest.raises(DatasetNotFoundError) as exc:
        load_credit_card_fraud(cfg.data)
    assert "kaggle" in exc.value.download_url.lower()
