"""Device utility — single entrypoint for all tensor-hosting decisions.

Per Execution Plan §1b.1: `get_device()` returns the resolved device
(config override → cuda if available → cpu). Every new nn.Module or
tensor allocation goes through here so nothing accidentally lands on CPU
when a GPU is present.

Also exposes:
  * `log_device_info(device)` — one-line structlog record with device name
    and free/total VRAM. Call once per training run.
  * `assert_on_gpu(module, name)` — fails loudly if a module whose config
    said "cuda" ended up on CPU. Used at Gate 1 acceptance.
  * `cuda_memory_snapshot()` — bytes on device, for MLflow logging per §1b.4.
  * `empty_cache()` — VRAM guard (§1b.5) between sweep cells.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import torch

from cadence.common.logging import get_logger

log = get_logger("cadence.common.device")


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    name: str
    total_vram_bytes: int
    free_vram_bytes: int
    torch_version: str
    cuda_version: str | None


_RESOLVED: torch.device | None = None


def get_device(override: str | None = None) -> torch.device:
    """Resolve the device once and cache it.

    Precedence: explicit `override` (e.g. "cuda", "cpu", "cuda:0") → cached
    resolution → cuda-if-available → cpu.

    A GPU-configured run that can't get one raises RuntimeError. Per §1b.5,
    we never silently fall back to CPU when the caller asked for cuda.
    """
    global _RESOLVED
    if override is not None:
        dev = torch.device(override)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"device override='{override}' requested cuda but "
                "torch.cuda.is_available() is False — refusing to silently "
                "fall back to CPU (see Execution Plan §1b.5)."
            )
        _RESOLVED = dev
        return dev
    if _RESOLVED is not None:
        return _RESOLVED
    _RESOLVED = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _RESOLVED


def reset_cache() -> None:
    """Clear the cached device — used by tests that toggle availability."""
    global _RESOLVED
    _RESOLVED = None


def describe(device: torch.device | None = None) -> DeviceInfo:
    """Structured snapshot of the device we're about to use."""
    dev = device if device is not None else get_device()
    if dev.type == "cuda":
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        free, total = torch.cuda.mem_get_info(idx)
        return DeviceInfo(
            device=dev,
            name=props.name,
            total_vram_bytes=int(total),
            free_vram_bytes=int(free),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
        )
    return DeviceInfo(
        device=dev,
        name="cpu",
        total_vram_bytes=0,
        free_vram_bytes=0,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )


def log_device_info(device: torch.device | None = None) -> DeviceInfo:
    """One-line structured record — call at the top of every training run."""
    info = describe(device)
    log.info(
        "device_info",
        device=str(info.device),
        name=info.name,
        vram_total_gb=round(info.total_vram_bytes / 1024**3, 3),
        vram_free_gb=round(info.free_vram_bytes / 1024**3, 3),
        torch_version=info.torch_version,
        cuda_version=info.cuda_version,
    )
    return info


def assert_on_gpu(module: torch.nn.Module, *, name: str = "module") -> None:
    """Refuse to proceed if a GPU-configured module ended up on CPU.

    Per Execution Plan §1b.4: a run that claims GPU must show bytes on the
    card. Call after `module.to(device)`.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(f"{name}: cuda not available but assert_on_gpu was called")
    params = list(module.parameters())
    if not params:
        raise RuntimeError(f"{name}: has no parameters — nothing to check")
    devs = {p.device.type for p in params}
    if devs != {"cuda"}:
        raise RuntimeError(
            f"{name}: expected all params on cuda, got devices={devs}. "
            "Refusing to silently fall back (Execution Plan §1b.5)."
        )


def cuda_memory_snapshot() -> dict[str, int]:
    """Bytes currently allocated + peak — logged to MLflow per §1b.4."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "max_allocated": 0, "reserved": 0}
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "max_allocated": int(torch.cuda.max_memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
    }


def reset_peak_memory() -> None:
    """Reset the peak-VRAM counter — call before each measured phase."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def empty_cache() -> None:
    """VRAM guard (§1b.5) — call between sweep cells."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_device(obj: Any, device: torch.device | None = None) -> Any:
    """Move a tensor / module / (tensor,tensor,...) tuple / dict onto device."""
    dev = device if device is not None else get_device()
    if isinstance(obj, torch.nn.Module):
        return obj.to(dev)
    if isinstance(obj, torch.Tensor):
        return obj.to(dev, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device(v, dev) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(to_device(v, dev) for v in obj)
    if isinstance(obj, list):
        return [to_device(v, dev) for v in obj]
    return obj
