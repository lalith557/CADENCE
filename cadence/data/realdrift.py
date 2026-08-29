"""Real-drift dataset loaders — Elec2 + Airlines.

For Step E's outcome-graded H2 (Execution Plan §3):
  * Elec2   — 45,312 half-hourly Australian electricity price observations
              1996-1998, binary target `UP/DOWN`. Loaded via `river.datasets.Elec2`.
  * Airlines — ~27k flight-delay records with genuine temporal drift.
              Loaded from `Dataset/Airlines1.arff` (the full-size file that
              ships in the repo; Airlines2-4 are subsamples).

Both loaders preserve the original row order — Step E's protocol is
"train on early slice, stream later slice," which requires the natural
timeline. `LoadedTemporalDataset` exposes the split boundary explicitly so
downstream harnesses don't have to guess it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LoadedTemporalDataset:
    name: str
    X: np.ndarray  # (n_rows, n_features), rows preserved in temporal order
    y: np.ndarray  # (n_rows,), int64
    feature_names: list[str]
    train_end_idx: int  # rows [0 : train_end_idx) are the "pretraining" slice
    task: Literal["binary_classification"] = "binary_classification"

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def train_slice(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X[: self.train_end_idx], self.y[: self.train_end_idx]

    def stream_slice(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X[self.train_end_idx :], self.y[self.train_end_idx :]

    def summary(self) -> dict:
        y_train = self.y[: self.train_end_idx]
        y_stream = self.y[self.train_end_idx :]
        return {
            "name": self.name,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "train_end_idx": self.train_end_idx,
            "positive_rate_train": float(np.mean(y_train)) if y_train.size else 0.0,
            "positive_rate_stream": float(np.mean(y_stream)) if y_stream.size else 0.0,
        }


def _standardize(X: np.ndarray) -> np.ndarray:
    """Column-wise standardize without leaking the stream slice's stats."""
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-6
    return ((X - mu) / sigma).astype(np.float32)


def load_elec2(
    *,
    train_frac: float = 0.30,
    standardize: bool = True,
    cache_dir: Path | None = None,
) -> LoadedTemporalDataset:
    """Elec2 via `river.datasets.Elec2()`.

    river caches the downloaded electricity.zip under `~/river_data/Elec2/`.
    First call may download ~700 KB.
    """
    try:
        from river.datasets import Elec2  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover — river is an optional extra
        raise ImportError(
            "river>=0.21 required for Elec2; install with `pip install river>=0.21`"
        ) from e

    ds = Elec2()
    rows_x: list[dict] = []
    rows_y: list[bool] = []
    for x, y in ds:
        rows_x.append(x)
        rows_y.append(bool(y))
    df = pd.DataFrame(rows_x)
    feature_names = list(df.columns)
    X = df.values.astype(np.float32)
    y = np.array(rows_y, dtype=np.int64)
    if standardize:
        X = _standardize(X)
    train_end = int(train_frac * X.shape[0])
    return LoadedTemporalDataset(
        name="elec2",
        X=X,
        y=y,
        feature_names=feature_names,
        train_end_idx=train_end,
    )


def load_airlines(
    path: Path | str = Path("Dataset/Airlines1.arff"),
    *,
    train_frac: float = 0.30,
    standardize: bool = True,
    max_rows: int | None = None,
) -> LoadedTemporalDataset:
    """Airlines delay dataset from a local ARFF file (Dataset/Airlines1.arff).

    Categorical columns (Airline, Flight, AirportFrom, AirportTo, DayOfWeek)
    are hash-encoded to a small integer feature so a plain MLP can consume
    them. This is *not* an ideal encoding for delay prediction — but for
    Step E we only care about drift-driven degradation of *whatever* model
    we train first, and hash encoding preserves the temporal order.
    """
    from scipy.io import arff  # noqa: PLC0415

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Airlines ARFF not found at {path}. "
            "Download from https://www.openml.org/search?type=data&id=1169 "
            "and save as Dataset/Airlines1.arff"
        )
    data, _meta = arff.loadarff(str(path))
    df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].dtype == object:
            # ARFF categoricals arrive as bytes; decode.
            df[col] = df[col].apply(
                lambda v: v.decode("utf-8") if isinstance(v, bytes) else v
            )

    target = "Delay"
    y = df[target].astype(int).values.astype(np.int64)

    numeric_cols = [c for c in df.columns if c != target and df[c].dtype != object]
    cat_cols = [c for c in df.columns if c != target and df[c].dtype == object]

    def _hash_col(col: pd.Series, n_bins: int = 64) -> np.ndarray:
        return col.map(lambda v: hash(v) % n_bins).values.astype(np.float32)

    numeric_X = df[numeric_cols].values.astype(np.float32) if numeric_cols else np.zeros(
        (len(df), 0), dtype=np.float32
    )
    cat_X = np.stack([_hash_col(df[c]) for c in cat_cols], axis=1) if cat_cols else np.zeros(
        (len(df), 0), dtype=np.float32
    )
    X = np.concatenate([numeric_X, cat_X], axis=1).astype(np.float32)
    feature_names = numeric_cols + [f"{c}_hash" for c in cat_cols]

    if max_rows is not None:
        X = X[:max_rows]
        y = y[:max_rows]

    if standardize:
        X = _standardize(X)

    train_end = int(train_frac * X.shape[0])
    return LoadedTemporalDataset(
        name="airlines",
        X=X,
        y=y,
        feature_names=feature_names,
        train_end_idx=train_end,
    )
