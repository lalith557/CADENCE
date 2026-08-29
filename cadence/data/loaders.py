"""Dataset loaders — one function per dataset used in the paper.

Each returns a `LoadedDataset` with numeric X_train/X_test/y_train/y_test
tensors as numpy arrays, plus feature-name metadata for the CDAG. Fixed splits
under a caller-provided seed.

All loaders raise `DatasetNotFoundError` (subclass of FileNotFoundError) with a
helpful message pointing to the original download URL if the local file is
missing — because Dataset/ is in .gitignore.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cadence.common.config import DataConfig


class DatasetNotFoundError(FileNotFoundError):
    def __init__(self, dataset_name: str, path: Path, download_url: str) -> None:
        super().__init__(
            f"[{dataset_name}] file not found at {path}. Download from {download_url} "
            f"and place it at that path."
        )
        self.dataset_name = dataset_name
        self.path = path
        self.download_url = download_url


@dataclass(frozen=True)
class LoadedDataset:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    task: Literal["binary_classification", "multiclass_classification"] = "binary_classification"

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_train": int(self.X_train.shape[0]),
            "n_test": int(self.X_test.shape[0]),
            "n_features": int(self.X_train.shape[1]),
            "positive_rate_train": float(np.mean(self.y_train)),
            "positive_rate_test": float(np.mean(self.y_test)),
            "task": self.task,
        }


def _split(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], name: str, cfg: DataConfig, seed: int
) -> LoadedDataset:
    stratify = y if len(np.unique(y)) == 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.test_size, random_state=seed, stratify=stratify
    )
    return LoadedDataset(
        name=name,
        X_train=X_tr.astype(np.float32),
        X_test=X_te.astype(np.float32),
        y_train=y_tr.astype(np.int64),
        y_test=y_te.astype(np.int64),
        feature_names=feature_names,
    )


def load_credit_card_fraud(cfg: DataConfig, seed: int = 42, *, scale: bool = True) -> LoadedDataset:
    """Kaggle Credit Card Fraud Detection — the neural adapter's headline dataset."""
    path = cfg.fraud_csv
    if not path.exists():
        raise DatasetNotFoundError(
            "credit_card_fraud",
            path,
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        )
    df = pd.read_csv(path)
    y = df["Class"].astype(np.int64).values
    X = df.drop(columns=["Class"]).values.astype(np.float32)
    feature_names = [c for c in df.columns if c != "Class"]

    if scale:
        # Time and Amount live on very different scales from V1..V28.
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)

    return _split(X, y, feature_names, "credit_card_fraud", cfg, seed)


def load_give_me_some_credit(cfg: DataConfig, seed: int = 42) -> LoadedDataset:
    """Kaggle Give Me Some Credit — the tree-ensemble adapter's headline dataset (Phase 6)."""
    path = cfg.gmsc_train_csv
    if not path.exists():
        raise DatasetNotFoundError(
            "give_me_some_credit",
            path,
            "https://www.kaggle.com/c/GiveMeSomeCredit",
        )
    df = pd.read_csv(path, index_col=0)
    # Impute the two commonly-missing columns with median so tree models can consume.
    for col in ("MonthlyIncome", "NumberOfDependents"):
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    y = df["SeriousDlqin2yrs"].astype(np.int64).values
    X = df.drop(columns=["SeriousDlqin2yrs"]).values.astype(np.float32)
    feature_names = [c for c in df.columns if c != "SeriousDlqin2yrs"]
    return _split(X, y, feature_names, "give_me_some_credit", cfg, seed)


def load_telco_churn(cfg: DataConfig, seed: int = 42) -> LoadedDataset:
    """Telco Customer Churn — the linear/logistic adapter."""
    path = cfg.telco_csv
    if not path.exists():
        raise DatasetNotFoundError(
            "telco_churn",
            path,
            "https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
        )
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"])
    # TotalCharges has stray blanks; coerce and impute with median.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    y = (df["Churn"] == "Yes").astype(np.int64).values
    df = df.drop(columns=["Churn"])
    df = pd.get_dummies(df, drop_first=True)
    feature_names = df.columns.tolist()
    X = df.values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    return _split(X, y, feature_names, "telco_churn", cfg, seed)


def load_adult(cfg: DataConfig, seed: int = 42) -> LoadedDataset:
    """UCI Adult — linear/logistic adapter alternative."""
    path = cfg.adult_dir / "adult.data"
    if not path.exists():
        raise DatasetNotFoundError("adult", path, "https://archive.ics.uci.edu/dataset/2/adult")
    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]
    df = pd.read_csv(path, header=None, names=columns, skipinitialspace=True, na_values=["?"])
    df = df.dropna()
    y = (df["income"] == ">50K").astype(np.int64).values
    df = df.drop(columns=["income"])
    df = pd.get_dummies(df, drop_first=True)
    feature_names = df.columns.tolist()
    X = df.values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    return _split(X, y, feature_names, "adult", cfg, seed)


def load_elec2(seed: int = 42) -> LoadedDataset:
    """Elec2 via river.datasets.Elec2 — real known-drift benchmark."""
    try:
        from river.datasets import Elec2
    except ImportError as e:
        raise ImportError(
            "river is required for load_elec2 — install via `pip install -e .[drift]`"
        ) from e

    rows = list(Elec2())  # returns (features_dict, label) tuples
    if not rows:
        raise RuntimeError("Elec2 stream returned no rows")
    feature_names = list(rows[0][0].keys())
    X = np.array([[r[0][k] for k in feature_names] for r in rows], dtype=np.float32)
    y = np.array([int(r[1]) for r in rows], dtype=np.int64)
    # Elec2 is a stream; use a temporal split (last 20% test) rather than random.
    split_idx = int(0.8 * len(X))
    return LoadedDataset(
        name="elec2",
        X_train=X[:split_idx],
        X_test=X[split_idx:],
        y_train=y[:split_idx],
        y_test=y[split_idx:],
        feature_names=feature_names,
    )


def load_mnist(cfg: DataConfig, seed: int = 42) -> LoadedDataset:
    """MNIST via torchvision, flattened for the continual-learning experiments."""
    try:
        from torchvision import datasets, transforms
    except ImportError as e:
        raise ImportError("torchvision is required for load_mnist — install torch") from e

    cache = cfg.torchvision_cache
    cache.mkdir(parents=True, exist_ok=True)
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(str(cache), train=True, download=True, transform=tfm)
    test = datasets.MNIST(str(cache), train=False, download=True, transform=tfm)

    X_train = train.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y_train = train.targets.numpy().astype(np.int64)
    X_test = test.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y_test = test.targets.numpy().astype(np.int64)
    feature_names = [f"px_{i}" for i in range(28 * 28)]
    return LoadedDataset(
        name="mnist",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        task="multiclass_classification",
    )


REGISTRY = {
    "credit_card_fraud": load_credit_card_fraud,
    "give_me_some_credit": load_give_me_some_credit,
    "telco_churn": load_telco_churn,
    "adult": load_adult,
}
