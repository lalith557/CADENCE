"""CADENCE executor — Part-3B EWC retrain + §4 fallbacks."""

from cadence.executor.escalation import (
    EscalationConfig,
    EscalationDecision,
    EscalationLadder,
    LadderAttempt,
    LadderResult,
)
from cadence.executor.executor import (
    ExecutorConfig,
    ExecutorEvent,
    ExecutorReport,
    RetrainExecutor,
)
from cadence.executor.fallbacks import (
    DiffuseFallbackConfig,
    FalseAlarmConfig,
    ResourceFallbackError,
    cold_start_scores,
    diffuse_responsibility_fallback,
    herfindahl_concentration,
    is_false_alarm,
    resource_fallback,
)

__all__ = [
    "DiffuseFallbackConfig",
    "EscalationConfig",
    "EscalationDecision",
    "EscalationLadder",
    "ExecutorConfig",
    "ExecutorEvent",
    "ExecutorReport",
    "FalseAlarmConfig",
    "LadderAttempt",
    "LadderResult",
    "ResourceFallbackError",
    "RetrainExecutor",
    "cold_start_scores",
    "diffuse_responsibility_fallback",
    "herfindahl_concentration",
    "is_false_alarm",
    "resource_fallback",
]
