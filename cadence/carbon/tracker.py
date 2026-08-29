"""CodeCarbon integration — measures real energy during actual training jobs.

Used by the executor and baseline runners to log the REAL cost/carbon of a
retrain, which we then compare against the RSO's PREDICTED cost from
`cadence.carbon.model.estimate_cost`. That calibration is part of what makes
the paper's H2 claim honest.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from cadence.common.logging import get_logger

log = get_logger("cadence.carbon.tracker")


@dataclass
class MeasuredCost:
    kg_co2: float
    energy_kwh: float
    wall_seconds: float
    source: str  # "codecarbon" | "fallback"


@contextmanager
def track_emissions(run_name: str) -> Iterator[MeasuredCost]:
    """Context manager. On exit, `result.kg_co2` etc. are populated.

    Falls back to a zero-filled MeasuredCost if CodeCarbon isn't installed —
    the sandbox model still provides the number for the reward.
    """
    result = MeasuredCost(kg_co2=0.0, energy_kwh=0.0, wall_seconds=0.0, source="fallback")
    try:
        from codecarbon import EmissionsTracker
    except ImportError:
        log.warning("codecarbon_not_installed", run=run_name)
        yield result
        return

    tracker = EmissionsTracker(
        project_name=f"cadence.{run_name}",
        save_to_file=False,
        log_level="error",
        allow_multiple_runs=True,
    )
    tracker.start()
    try:
        yield result
    finally:
        emissions = tracker.stop() or 0.0
        result.kg_co2 = float(emissions)
        result.energy_kwh = float(getattr(tracker, "_total_energy", 0.0) or 0.0)
        result.wall_seconds = float(getattr(tracker, "_last_measured_time", 0.0) or 0.0)
        result.source = "codecarbon"
