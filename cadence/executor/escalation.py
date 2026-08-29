"""Escalation ladder (§4.3) + unrecoverable-drift guard (§4.5).

§4.3 escalation ladder:
    partial retrain fails (rollback fired)
      -> try full retrain
        -> if that fails too, flag a human, DO NOT silently serve.

§4.5 unrecoverable-drift guard:
    If a scenario has flipped the concept beyond what any retrain on old
    labels can fix (Part 6's v14 concept flip / R-Gate-C's calibration gap),
    we should stop burning compute after N consecutive rollbacks and
    escalate to a human. This turns R-4's "v14 unrecoverable for every
    strategy" finding into a feature.

Both live above the executor: they call `RetrainExecutor.execute(...)` in a
loop, watch the returned `ExecutorReport`, and decide the next attempt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from cadence.common.logging import get_logger
from cadence.executor.executor import ExecutorReport, RetrainExecutor

log = get_logger("cadence.executor.escalation")


class EscalationDecision(str, Enum):
    PROMOTED = "promoted"  # some attempt succeeded — model updated
    ESCALATED_HUMAN = "escalated_human"  # all attempts failed, alert operator
    NO_OP = "no_op"  # nothing worth trying (empty pipeline)


@dataclass
class EscalationConfig:
    """§4.3 + §4.5 knobs."""

    # §4.3: order of actions to try, top-down. Empty list = no retrain.
    ladder: tuple[str, ...] = ("partial", "full")
    # §4.5: after this many CONSECUTIVE full-ladder failures (i.e. the full
    # ladder rolled back top-to-bottom), flag as unrecoverable.
    unrecoverable_streak: int = 3


@dataclass
class LadderAttempt:
    action: str
    was_rolled_back: bool
    post_f1: float
    shadow_f1: float


@dataclass
class LadderResult:
    decision: EscalationDecision
    attempts: list[LadderAttempt] = field(default_factory=list)
    reports: list[ExecutorReport] = field(default_factory=list)
    unrecoverable_streak: int = 0
    escalation_reason: str = ""


class EscalationLadder:
    """Runs the §4.3 ladder against a RetrainExecutor.

    Tracks a running "unrecoverable streak" — the number of consecutive
    calls to `run_ladder` that ended with EVERY attempt rolling back. When
    the streak hits `cfg.unrecoverable_streak`, subsequent calls short-
    circuit to ESCALATED_HUMAN without spending compute (§4.5).
    """

    def __init__(self, cfg: EscalationConfig | None = None):
        self.cfg = cfg or EscalationConfig()
        self._streak = 0
        self._is_unrecoverable = False

    @property
    def is_unrecoverable(self) -> bool:
        return self._is_unrecoverable

    @property
    def current_streak(self) -> int:
        return self._streak

    def reset_streak(self) -> None:
        self._streak = 0
        self._is_unrecoverable = False

    def run_ladder(
        self,
        executor: RetrainExecutor,
        *,
        window_X,
        window_y,
        target_layer_for_partial: str | None,
        predicted_post_f1: float | None = None,
    ) -> LadderResult:
        """§4.3: try each rung of `cfg.ladder`; stop when one promotes.

        §4.5: if the unrecoverable-streak threshold has already fired, do
        nothing and return ESCALATED_HUMAN without burning compute.
        """
        if self._is_unrecoverable:
            log.warning(
                "unrecoverable_shortcircuit",
                streak=self._streak,
                threshold=self.cfg.unrecoverable_streak,
            )
            return LadderResult(
                decision=EscalationDecision.ESCALATED_HUMAN,
                escalation_reason=(
                    f"unrecoverable streak {self._streak} >= "
                    f"threshold {self.cfg.unrecoverable_streak}; skipping retrain"
                ),
            )

        attempts: list[LadderAttempt] = []
        reports: list[ExecutorReport] = []

        for action in self.cfg.ladder:
            tgt = target_layer_for_partial if action == "partial" else None
            report = executor.execute(
                action=action,
                window_X=window_X,
                window_y=window_y,
                target_layer=tgt,
                predicted_post_f1=predicted_post_f1,
            )
            attempts.append(
                LadderAttempt(
                    action=action,
                    was_rolled_back=report.event.was_rolled_back,
                    post_f1=report.event.post_f1,
                    shadow_f1=report.event.shadow_f1,
                )
            )
            reports.append(report)
            if not report.event.was_rolled_back:
                # Something promoted — reset streak and return.
                log.info(
                    "ladder_promoted",
                    action=action,
                    n_attempts=len(attempts),
                )
                self._streak = 0
                return LadderResult(
                    decision=EscalationDecision.PROMOTED,
                    attempts=attempts,
                    reports=reports,
                    unrecoverable_streak=self._streak,
                )

        # Every rung rolled back.
        self._streak += 1
        log.warning(
            "ladder_all_failed",
            n_attempts=len(attempts),
            streak=self._streak,
            threshold=self.cfg.unrecoverable_streak,
        )
        if self._streak >= self.cfg.unrecoverable_streak:
            self._is_unrecoverable = True
            log.error(
                "unrecoverable_drift_flagged",
                streak=self._streak,
                threshold=self.cfg.unrecoverable_streak,
            )
        return LadderResult(
            decision=EscalationDecision.ESCALATED_HUMAN,
            attempts=attempts,
            reports=reports,
            unrecoverable_streak=self._streak,
            escalation_reason=(
                f"all ladder rungs rolled back; streak={self._streak}"
                if not self._is_unrecoverable
                else f"unrecoverable drift flagged after {self._streak} failures"
            ),
        )
