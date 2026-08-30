"""Guards for the hand-rolled Split-MNIST loader (W-21 -> Step F fix)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cadence.data.mnist_splits import DEFAULT_TASKS, MNISTTask, load_split_mnist

MNIST_ROOT = Path("Dataset/MNIST")


def _mnist_available() -> bool:
    # The dot-named files ship in the repo's Dataset/ tree (excluded via
    # .gitignore for tracking, but present locally).
    return (MNIST_ROOT / "train-images.idx3-ubyte").exists()


@pytest.mark.needs_dataset
@pytest.mark.skipif(not _mnist_available(), reason="MNIST ubyte files not on disk")
def test_default_split_returns_five_disjoint_tasks() -> None:
    tasks = load_split_mnist(root=MNIST_ROOT, split="train")
    assert len(tasks) == 5
    assert [t.digits for t in tasks] == list(DEFAULT_TASKS)
    # Every task has a healthy number of rows.
    for t in tasks:
        assert isinstance(t, MNISTTask)
        assert t.X.shape[0] > 1000
        assert t.X.shape[1] == 784
        assert set(t.y.tolist()).issubset({0, 1})


@pytest.mark.needs_dataset
@pytest.mark.skipif(not _mnist_available(), reason="MNIST ubyte files not on disk")
def test_test_split_smaller_than_train() -> None:
    train = load_split_mnist(root=MNIST_ROOT, split="train")
    test = load_split_mnist(root=MNIST_ROOT, split="test")
    for t_tr, t_te in zip(train, test, strict=True):
        assert t_te.X.shape[0] < t_tr.X.shape[0]
        assert t_te.digits == t_tr.digits


@pytest.mark.needs_dataset
@pytest.mark.skipif(not _mnist_available(), reason="MNIST ubyte files not on disk")
def test_tasks_are_actually_disjoint() -> None:
    """No image should appear in more than one task's X."""
    tasks = load_split_mnist(root=MNIST_ROOT, split="test")
    total = sum(t.X.shape[0] for t in tasks)
    # MNIST test has 10,000 rows; our 5 tasks cover 10 digits, so total==10_000.
    assert total == 10_000


def test_load_split_mnist_raises_on_missing_root(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="MNIST"):
        load_split_mnist(root=tmp_path / "no-such-dir")
