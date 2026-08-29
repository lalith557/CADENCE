"""Compute + carbon cost model for the RSO reward.

Two sources:
  1. **Approximate model** (this module) — closed-form: FLOPs → GPU-seconds
     under a nominal throughput, then GPU-seconds × grid carbon intensity.
     Deterministic, used inside the Gym sandbox where CodeCarbon can't run
     during PPO's millions of simulated steps.

  2. **CodeCarbon integration** (cadence.carbon.tracker) — wraps real training
     jobs to log measured energy + emissions to MLflow.

Both must be reported so the paper can honestly compare the RSO's *predicted*
cost signal against measured cost.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostEstimate:
    gpu_seconds: float
    energy_kwh: float
    kg_co2: float


@dataclass
class HardwareProfile:
    """Nominal throughput for the sandbox.

    Default = RTX 4070 Laptop GPU (the actual machine — Execution Plan §1b bullet 3
    fixed the previous gtx-1650 default that fed a wrong FLOPs→GPU-seconds→CO2
    signal into the RSO reward). Override by passing a different profile.

    tflops_fp32 source: techpowerup GeForce RTX 4070 Laptop (7168 CUDA cores,
    ~15.6 TFLOPs FP32 at boost). TDP 115 W in the Laptop variant.
    """

    device_name: str = "rtx-4070-laptop"
    tflops_fp32: float = 15.62
    tdp_watts: float = 115.0
    utilization: float = 0.4  # realistic average during small-model training
    baseline_watts: float = 25.0  # host draw when idle


# Retained for reproducing the older R-1..R-4 numbers on demand.
GTX_1650_PROFILE = HardwareProfile(
    device_name="gtx-1650",
    tflops_fp32=2.98,
    tdp_watts=75.0,
    utilization=0.4,
    baseline_watts=20.0,
)


def detect_hardware_profile() -> HardwareProfile:
    """Best-effort hardware detection.

    Uses `torch.cuda.get_device_name(0)` when CUDA is available. Falls back
    to the RTX 4070 Laptop default (this machine) so the reward stays sane
    even if detection fails.
    """
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            if "4070" in name and "laptop" in name:
                return HardwareProfile()  # default already targets this
            if "1650" in name:
                return GTX_1650_PROFILE
            # Unknown card — keep the default (RTX 4070 Laptop) but relabel
            # so results.md shows what actually ran.
            profile = HardwareProfile()
            profile.device_name = name
            return profile
    except Exception:  # noqa: BLE001 — best-effort only
        pass
    return HardwareProfile()


@dataclass
class GridProfile:
    """Grid carbon intensity in gCO2e per kWh. Default = global average per Ember 2024."""

    region: str = "global-avg"
    intensity_g_per_kwh: float = 481.0


def estimate_cost(
    flops_per_forward: int,
    n_samples: int,
    epochs: int,
    *,
    grad_pass_multiplier: float = 3.0,
    hardware: HardwareProfile = HardwareProfile(),
    grid: GridProfile = GridProfile(),
) -> CostEstimate:
    """Deterministic cost + carbon estimate for a training run.

    `grad_pass_multiplier`: backward+optimizer is ~2-3x forward FLOPs — use 3.
    """
    total_flops = flops_per_forward * n_samples * epochs * grad_pass_multiplier
    device_flops_per_s = hardware.tflops_fp32 * 1e12 * hardware.utilization
    gpu_seconds = total_flops / max(device_flops_per_s, 1.0)
    watts_effective = hardware.baseline_watts + hardware.tdp_watts * hardware.utilization
    energy_kwh = (watts_effective * gpu_seconds) / 3_600_000.0
    kg_co2 = energy_kwh * grid.intensity_g_per_kwh / 1000.0
    return CostEstimate(gpu_seconds=gpu_seconds, energy_kwh=energy_kwh, kg_co2=kg_co2)
