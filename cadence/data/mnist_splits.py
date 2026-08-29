"""Split-MNIST for a real H3 forgetting test (fixes Step D's single-task
limitation without needing the full Avalanche library).

The 10 digit classes get partitioned into 5 disjoint binary tasks:
    Task 0: digits {0,1}   → label 0 / 1
    Task 1: digits {2,3}   → label 0 / 1
    Task 2: digits {4,5}   → label 0 / 1
    Task 3: digits {6,7}   → label 0 / 1
    Task 4: digits {8,9}   → label 0 / 1

For the H3 test we pretrain on Task 0, then run partial or full retrain
on Task 1, and measure forgetting = pre_task0_F1 - post_task0_F1. Task 0
and Task 1 are *genuinely* disjoint — a Task-1 image is a 2 or 3, and
the model never sees a 2 or 3 during pretraining.

We load from the raw ubyte files that ship in Dataset/MNIST/ so no
network is required at run time (the R-Gate-F run must be reproducible
offline).
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MNISTTask:
    task_id: int
    digits: tuple[int, int]
    X: np.ndarray  # (n, 784) float32 in [0, 1]
    y: np.ndarray  # (n,) int64 in {0, 1}

    def summary(self) -> dict:
        return {
            "task_id": self.task_id,
            "digits": list(self.digits),
            "n": int(self.X.shape[0]),
            "positive_rate": float(np.mean(self.y)),
        }


def _read_ubyte_images(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path}: bad magic {magic}")
        buf = f.read(n * rows * cols)
    imgs = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols)
    return imgs.astype(np.float32) / 255.0


def _read_ubyte_labels(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path}: bad magic {magic}")
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


DEFAULT_TASKS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
)


def load_split_mnist(
    root: Path | str = Path("Dataset/MNIST"),
    *,
    tasks: tuple[tuple[int, int], ...] = DEFAULT_TASKS,
    split: str = "train",
) -> list[MNISTTask]:
    """Load MNIST + partition into `tasks` disjoint binary tasks.

    `split` = "train" or "test".
    """
    root = Path(root)

    def _resolve(base: str) -> Path:
        """MNIST is packaged inconsistently — try dots first (regular files)
        then dashes (in some downloads these are directories, not files)."""
        # Some torchvision downloads produce `foo.idx3-ubyte`, others
        # `foo-idx3-ubyte`. On this machine the dot-versions are actual
        # files and the dash-versions are stub directories.
        candidates = [
            root / f"{base}.idx{'3' if 'images' in base else '1'}-ubyte",
            root / f"{base}-idx{'3' if 'images' in base else '1'}-ubyte",
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c
        raise FileNotFoundError(
            f"MNIST {base} not found in {root}; tried {candidates}. Download from "
            "http://yann.lecun.com/exdb/mnist/ and unpack to Dataset/MNIST/"
        )

    if split == "train":
        img_path = _resolve("train-images")
        lbl_path = _resolve("train-labels")
    else:
        img_path = _resolve("t10k-images")
        lbl_path = _resolve("t10k-labels")

    X = _read_ubyte_images(img_path)
    y = _read_ubyte_labels(lbl_path)

    out: list[MNISTTask] = []
    for tid, (d0, d1) in enumerate(tasks):
        mask = (y == d0) | (y == d1)
        Xt = X[mask]
        yt = (y[mask] == d1).astype(np.int64)  # relabel: d0 -> 0, d1 -> 1
        out.append(MNISTTask(task_id=tid, digits=(d0, d1), X=Xt, y=yt))
    return out
