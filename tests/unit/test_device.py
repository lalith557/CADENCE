"""Guards for cadence.common.device.

These tests exist because the Execution Plan §1b.5 explicitly forbids silent
CPU fallback — the whole point of the helper is that a run that asked for
cuda either gets cuda or fails loudly.
"""

from __future__ import annotations

import pytest
import torch

from cadence.common.device import (
    assert_on_gpu,
    cuda_memory_snapshot,
    describe,
    empty_cache,
    get_device,
    log_device_info,
    reset_cache,
    to_device,
)


def test_get_device_auto_returns_valid_device() -> None:
    reset_cache()
    dev = get_device()
    assert dev.type in {"cuda", "cpu"}
    # cached
    assert get_device() is dev or get_device() == dev


def test_get_device_override_cpu_always_works() -> None:
    reset_cache()
    dev = get_device("cpu")
    assert dev == torch.device("cpu")


def test_get_device_override_cuda_fails_loudly_when_unavailable(monkeypatch) -> None:
    reset_cache()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="silent"):
        get_device("cuda")


def test_describe_returns_device_info_on_cpu() -> None:
    reset_cache()
    info = describe(torch.device("cpu"))
    assert info.device.type == "cpu"
    assert info.total_vram_bytes == 0
    assert info.torch_version


def test_log_device_info_smoke() -> None:
    reset_cache()
    info = log_device_info(torch.device("cpu"))
    assert info.name == "cpu"


def test_cuda_memory_snapshot_returns_expected_keys() -> None:
    snap = cuda_memory_snapshot()
    assert set(snap.keys()) == {"allocated", "max_allocated", "reserved"}


def test_empty_cache_is_safe_without_cuda() -> None:
    # Should not raise even on a CPU-only environment.
    empty_cache()


def test_to_device_handles_tensor_dict_and_module() -> None:
    reset_cache()
    dev = torch.device("cpu")
    t = torch.zeros(3)
    assert to_device(t, dev).device == dev
    d = {"a": torch.zeros(2), "b": torch.zeros(2)}
    d2 = to_device(d, dev)
    assert all(v.device == dev for v in d2.values())
    m = torch.nn.Linear(2, 2)
    m2 = to_device(m, dev)
    assert next(m2.parameters()).device == dev


@pytest.mark.gpu
def test_assert_on_gpu_when_cuda_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    m = torch.nn.Linear(4, 4).to("cuda")
    assert_on_gpu(m, name="linear")  # must not raise


@pytest.mark.gpu
def test_assert_on_gpu_rejects_cpu_module_when_cuda_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    m = torch.nn.Linear(4, 4)  # stays on CPU
    with pytest.raises(RuntimeError, match="expected all params on cuda"):
        assert_on_gpu(m, name="cpu_linear")
