#!/usr/bin/env python
"""Evaluate the 8 Stage-1 sanity gates from docs/gate_a_preregistration.md Part 1.

Reads:
  * experiments/gate_a_ledger.json — per-seed status + wall seconds
  * logs/stage1_w38fix.log         — dual_update + action + reward events
  * MLflow experiment `cadence-gate-a` (sqlite:///experiments/mlflow.db) — PPO
    training curves (entropy, loss, reward) [G1-G5 TODO markers if unavailable]

Prints per-gate PASS / FAIL / TBD with the numeric evidence, and returns
exit 0 iff all applicable gates PASS. Intended workflow:
    # Once Stage 1's ledger shows all 3 seeds COMPLETED
    python scripts/evaluate_stage1_gates.py

Writes a machine-readable report at experiments/gate_a_stage1_gates.json for
the follow-up R-Gate-A-stage1 entry in docs/results.md.

Bootstrap policy:
  * If any gate FAILs -> exit 1 (do NOT proceed to Stage 2).
  * If all gates PASS OR every failing gate is TBD -> exit 0.
  * TBD gates get printed with their exact data source so a human can close
    them by hand.

Honesty:
  * Never claims PASS on an INTERRUPTED/FAILED/NOT_STARTED seed.
  * Never claims PASS on a gate whose data source is missing.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


LEDGER_PATH = Path("experiments/gate_a_ledger.json")
DEFAULT_LOG = Path("logs/stage1_w38fix.log")
DEFAULT_MLFLOW_URI = "sqlite:///experiments/mlflow.db"
REPORT_PATH = Path("experiments/gate_a_stage1_gates.json")


@dataclass
class GateResult:
    name: str
    verdict: str  # PASS / FAIL / TBD / NA
    threshold: str
    measured: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.name,
            "verdict": self.verdict,
            "threshold": self.threshold,
            "measured": self.measured,
            "evidence": self.evidence,
        }


DUAL_MARKER = "dual_update"
LAMBDA_RE = re.compile(r"lambda_new=([0-9.eE+-]+)")
VIOLATION_RE = re.compile(r"episode_mean_violation=([0-9.eE+-]+)")


def parse_dual_updates(log_path: Path) -> list[tuple[float, float]]:
    """Return list of (lambda_new, episode_mean_violation) events, in order.

    Extracts the two fields independently — the log's structlog-style ordering
    is alphabetical (ema_violation, episode_mean_violation, lambda_new), so a
    single positional regex fails.
    """
    if not log_path.exists():
        return []
    updates: list[tuple[float, float]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if DUAL_MARKER not in line:
                continue
            lm = LAMBDA_RE.search(line)
            vm = VIOLATION_RE.search(line)
            if not lm or not vm:
                continue
            try:
                updates.append((float(lm.group(1)), float(vm.group(1))))
            except ValueError:
                continue
    return updates


def load_ledger(path: Path = LEDGER_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _gate_g6_lambda_max(dual_updates: list[tuple[float, float]]) -> GateResult:
    """G6: final lambda <= 50 (no runaway toward always-retrain)."""
    thresh = "final lambda <= 50 on last 30% of training"
    if not dual_updates:
        return GateResult(
            "G6",
            "TBD",
            thresh,
            "no dual_update events found in log",
            evidence={"log_events_found": 0},
        )
    n = len(dual_updates)
    last_30_pct = dual_updates[int(0.7 * n) :]
    if not last_30_pct:
        last_30_pct = dual_updates[-max(1, n // 3) :]
    last_lambdas = [u[0] for u in last_30_pct]
    final_lambda = last_lambdas[-1]
    mean_last_lambda = statistics.fmean(last_lambdas)
    passes = mean_last_lambda <= 50.0
    return GateResult(
        "G6",
        "PASS" if passes else "FAIL",
        thresh,
        f"mean lambda over last 30% = {mean_last_lambda:.2f}; final = {final_lambda:.2f}",
        evidence={
            "n_dual_updates": n,
            "final_lambda": final_lambda,
            "mean_last_30pct_lambda": mean_last_lambda,
            "max_lambda_ever": max(u[0] for u in dual_updates),
        },
    )


def _gate_g7_cross_seed_variance(ledger: dict[str, Any]) -> GateResult:
    """G7: std of final-30%-mean reward across seeds >= 0.02.

    Uses per-seed `metrics.rso_mean_f1` per scenario (proxy for return until
    MLflow episode returns are pulled), taken as the mean across scenarios.
    """
    thresh = "std of mean_f1 across seeds >= 0.02"
    completed = [
        s for s in ledger.get("stages", {}).get("1", {}).get("seeds", [])
        if s.get("status") == "COMPLETED" and s.get("metrics")
    ]
    if len(completed) < 2:
        return GateResult(
            "G7",
            "TBD",
            thresh,
            f"only {len(completed)} COMPLETED seed(s) with metrics; need >=2",
            evidence={"completed_seeds": len(completed)},
        )
    seed_means: list[float] = []
    for entry in completed:
        f1s = [v.get("rso_mean_f1") for v in entry["metrics"].values()
               if isinstance(v, dict) and v.get("rso_mean_f1") is not None]
        if f1s:
            seed_means.append(statistics.fmean(f1s))
    if len(seed_means) < 2:
        return GateResult(
            "G7", "TBD", thresh,
            f"only {len(seed_means)} seed(s) have rso_mean_f1",
            evidence={"seed_means": seed_means},
        )
    std = statistics.pstdev(seed_means)
    passes = std >= 0.02
    return GateResult(
        "G7",
        "PASS" if passes else "FAIL",
        thresh,
        f"cross-seed std = {std:.4f} over {len(seed_means)} seeds",
        evidence={"seed_means": seed_means, "std": std},
    )


def _gate_g8_budget(ledger: dict[str, Any]) -> GateResult:
    """G8: Stage 1 completes within 4h wall on RTX 4070 Laptop, no crashes."""
    thresh = "all 3 seeds COMPLETED; total wall <= 4h"
    seeds = ledger.get("stages", {}).get("1", {}).get("seeds", [])
    if not seeds:
        return GateResult("G8", "TBD", thresh, "ledger has no seeds yet", evidence={})

    completed = [s for s in seeds if s.get("status") == "COMPLETED"]
    non_completed = [s for s in seeds if s.get("status") != "COMPLETED"]
    total_wall = sum(s.get("wall_seconds", 0) or 0 for s in seeds)
    hours = total_wall / 3600.0

    if len(completed) < len(seeds):
        return GateResult(
            "G8",
            "TBD",
            thresh,
            f"{len(completed)}/{len(seeds)} seeds COMPLETED; "
            f"partial wall so far = {hours:.2f}h; open statuses = "
            f"{[s.get('status') for s in non_completed]}",
            evidence={
                "completed": len(completed),
                "total_seeds": len(seeds),
                "partial_wall_hours": hours,
            },
        )
    # All completed — evaluate against 4h.
    passes = hours <= 4.0 and all(s.get("status") == "COMPLETED" for s in seeds)
    return GateResult(
        "G8",
        "PASS" if passes else "FAIL",
        thresh,
        f"total wall = {hours:.2f}h across {len(seeds)} COMPLETED seeds",
        evidence={"wall_hours": hours, "per_seed_wall_s": [s.get("wall_seconds") for s in seeds]},
    )


def _gate_g_mlflow_stub(gate_name: str, description: str) -> GateResult:
    return GateResult(
        gate_name,
        "TBD",
        description,
        f"pending MLflow query — connect to `{DEFAULT_MLFLOW_URI}` "
        f"experiment `cadence-gate-a`, filter runs by tag stage=1, and read "
        f"metrics from ppo/entropy_loss (G2), ppo/ep_rew_mean (G3), and the "
        f"per-step info dict components delta_f1/cost_penalty/sla_penalty "
        f"(G4/G5) or action distribution (G1).",
    )


def evaluate(log_path: Path = DEFAULT_LOG) -> list[GateResult]:
    ledger = load_ledger()
    dual_updates = parse_dual_updates(log_path)

    results: list[GateResult] = []

    # G1-G5: pull from MLflow. Stubbed with TBD + data source for now.
    results.append(_gate_g_mlflow_stub(
        "G1", "no action collapse: max action_pct on last 30% of training <= 85%"
    ))
    results.append(_gate_g_mlflow_stub(
        "G2", "mean policy entropy over last 30% of training >= 0.15 nats"
    ))
    results.append(_gate_g_mlflow_stub(
        "G3", "rolling mean ep_return (last 30%) > (first 30%) by >= 0.10, not just SLA-term-driven"
    ))
    results.append(_gate_g_mlflow_stub(
        "G4", "|cost_penalty_component| avg >= 1% of |delta_f1_component| avg per step"
    ))
    results.append(_gate_g_mlflow_stub(
        "G5", "sla_penalty AND cost_penalty both shrink over training (not always-retrain gaming)"
    ))

    # G6-G8: computable from log + ledger directly.
    results.append(_gate_g6_lambda_max(dual_updates))
    if ledger is None:
        results.append(GateResult("G7", "TBD", "cross-seed reward std >= 0.02",
                                  "ledger missing", evidence={}))
        results.append(GateResult("G8", "TBD", "3 seeds COMPLETED, wall <= 4h",
                                  "ledger missing", evidence={}))
    else:
        results.append(_gate_g7_cross_seed_variance(ledger))
        results.append(_gate_g8_budget(ledger))
    return results


def print_report(results: list[GateResult]) -> None:
    verdict_counts: dict[str, int] = {}
    print("\n=== Stage 1 Gate Evaluation (per docs/gate_a_preregistration.md Part 1) ===\n")
    for r in results:
        verdict_counts[r.verdict] = verdict_counts.get(r.verdict, 0) + 1
        print(f"[{r.verdict:<4}] {r.name}: {r.threshold}")
        print(f"        measured: {r.measured}")
        if r.evidence:
            for k, v in r.evidence.items():
                if isinstance(v, float):
                    print(f"          {k}: {v:.4f}")
                elif isinstance(v, list) and len(v) <= 5:
                    print(f"          {k}: {v}")
                else:
                    print(f"          {k}: {v}")
        print()
    print("--- Summary ---")
    for v in ("PASS", "FAIL", "TBD", "NA"):
        if verdict_counts.get(v):
            print(f"  {v}: {verdict_counts[v]}")
    print()
    if verdict_counts.get("FAIL"):
        print("!!! At least one gate FAILED. DO NOT proceed to Stage 2. Diagnose per")
        print("    pre-registration Part 1's single-fix loop; restart Stage 1 clean.")
    elif verdict_counts.get("TBD"):
        print("Some gates are TBD — close them by querying MLflow per each gate's")
        print("measured-note. Only proceed to Stage 2 when every gate is PASS.")
    else:
        print(">>> All applicable gates PASS. Freeze method; hand off to Stage 2.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG,
                        help=f"Stage 1 log path (default: {DEFAULT_LOG})")
    parser.add_argument("--report-out", type=Path, default=REPORT_PATH,
                        help=f"Where to write JSON report (default: {REPORT_PATH})")
    args = parser.parse_args(argv)

    results = evaluate(log_path=args.log)
    print_report(results)

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with args.report_out.open("w", encoding="utf-8") as f:
        json.dump({
            "gates": [r.as_dict() for r in results],
            "log_path": str(args.log),
            "ledger_path": str(LEDGER_PATH),
            "mlflow_uri": DEFAULT_MLFLOW_URI,
        }, f, indent=2, default=str)
    print(f"Report written to {args.report_out}")

    if any(r.verdict == "FAIL" for r in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
