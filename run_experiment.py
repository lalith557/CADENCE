#!/usr/bin/env python
"""Gate A resumable runner — one command, survives laptop interruption.

Wraps `benchmarks.phase_a_run` per-seed with an atomic ledger so a killed
session never loses more than one seed. Implements the two-stage protocol
frozen in `docs/gate_a_preregistration.md`.

Stage 1 (diagnostic — pre-reg Part 1):
    3 seeds × 3000 timesteps × 3 contested scenarios, six W-32..W-37 fixes on.
    Purpose: gate G1-G8 must pass before Stage 2 runs.

Stage 2 (confirmatory — pre-reg Part 2):
    10 seeds × 30000 timesteps × 8 scenarios (5 default + 3 contested).
    Purpose: produce the pre-registered STRONG / PRACTICAL / INCONCLUSIVE /
    NO-ADVANTAGE verdict.

Usage:
    python run_experiment.py --stage 1                    # start / continue Stage 1
    python run_experiment.py --stage 2 --resume           # continue Stage 2
    python run_experiment.py --stage 1 --seed 2           # rerun one seed
    python run_experiment.py --stage 1 --dry-run          # print plan, no training

Ledger: experiments/gate_a_ledger.json (atomic writes, one entry per seed).
Per-seed result: experiments/gate_a_stage{1,2}/seed_{N}.json (from phase_a_run).
Per-seed PPO checkpoint: experiments/rso_ppo_phase_a_seed{N}.zip (from phase_a_run).

A kill (Ctrl-C) marks the in-flight seed INTERRUPTED and prints the exact
resume command. A killed seed is INTERRUPTED, never PASS/FAIL.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LEDGER_PATH = Path("experiments/gate_a_ledger.json")


# Frozen stage parameters. Modifying these invalidates the pre-registration —
# treat as read-only under a signed pre-reg.
STAGE_PARAMS: dict[int, dict[str, Any]] = {
    1: {
        "seeds": 3,
        "train_timesteps": 3000,
        "n_windows": 3,
        "window_size": 1024,
        "sla": 0.65,
        "contested_only": True,
        # W-42: Stage-1 diagnostic is "No baselines" per pre-reg Part 1.
        "skip_baselines": True,
        # W-44 REVERTED (fail-4): eval_window_size=4096 drove G8 to 10.06h and
        # its intended G7 benefit was never fairly tested (W-43's policy collapse
        # masked it in the same confounded run). Back to the training window
        # (override off). The --eval-window-size flag remains available for a
        # future *clean* G7 re-test at the healthy cost_scale=1e7 config, at a
        # more modest size (e.g. 2048) so G8 stays within budget.
        # "eval_window_size": 2048,  # <- re-enable for a clean G7 re-test
    },
    2: {
        "seeds": 10,
        "train_timesteps": 30000,
        "n_windows": 15,
        "window_size": 2048,
        "sla": 0.65,
        "contested_only": False,
        # Stage-2 confirmatory DOES compare against baselines.
        "skip_baselines": False,
    },
}


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def config_hash(config_path: Path) -> str:
    try:
        from cadence.common.config import load_config

        return load_config(config_path).hash()
    except Exception:
        return "unknown"


def atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON via temp file + os.replace so a kill mid-write cannot corrupt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup; re-raise so caller sees the failure.
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_ledger() -> dict[str, Any] | None:
    if not LEDGER_PATH.exists():
        return None
    with LEDGER_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def init_ledger(config_path: Path, git_val: str, cfg_val: str) -> dict[str, Any]:
    return {
        "experiment_id": f"gate-a-{git_val[:8]}-{cfg_val}",
        "git_commit": git_val,
        "config_hash": cfg_val,
        "config_path": str(config_path),
        "created": utcnow_iso(),
        "stages": {},
    }


def init_stage(ledger: dict[str, Any], stage: int, params: dict[str, Any]) -> None:
    key = str(stage)
    if key not in ledger["stages"]:
        ledger["stages"][key] = {
            "params": params,
            "started": utcnow_iso(),
            "seeds": [{"seed": s, "status": "NOT_STARTED"} for s in range(params["seeds"])],
        }


def resume_command(config_path: Path, stage: int) -> str:
    return f"python run_experiment.py --config {config_path} --stage {stage} --resume"


def extract_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Pull a compact summary of RSO's per-scenario numbers for the ledger."""
    metrics: dict[str, Any] = {}
    scenarios = result.get("scenarios", {})
    for scen_name, entry in scenarios.items():
        aggs = entry.get("aggregates", {}) or {}
        rso = aggs.get("rso_ppo", {}) or {}
        mean_f1 = rso.get("mean_f1") or [None, None]
        gpu_hr = rso.get("gpu_hr") or [None, None]
        metrics[scen_name] = {
            "rso_mean_f1": mean_f1[0] if isinstance(mean_f1, list) else None,
            "rso_gpu_hr": gpu_hr[0] if isinstance(gpu_hr, list) else None,
        }
    return metrics


def _default_run_one_seed(
    config_path: Path, stage: int, seed: int, params: dict[str, Any]
) -> tuple[int, Path, float]:
    """Launch phase_a_run for one seed via subprocess. Returns (rc, out_path, wall)."""
    out_dir = Path(f"experiments/gate_a_stage{stage}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_{seed}.json"

    cmd = [
        sys.executable,
        "-m",
        "benchmarks.phase_a_run",
        "--config",
        str(config_path),
        "--only-seed",
        str(seed),
        "--seeds",
        "1",  # loop scope placeholder; --only-seed narrows the actual iteration
        "--train-timesteps",
        str(params["train_timesteps"]),
        "--n-windows",
        str(params["n_windows"]),
        "--window-size",
        str(params["window_size"]),
        "--sla",
        str(params["sla"]),
        "--per-seed-policies",
        "--out",
        str(out_path),
    ]
    if params["contested_only"]:
        cmd.append("--contested-only")
    if params.get("skip_baselines"):
        cmd.append("--skip-baselines")
    if params.get("eval_window_size"):
        cmd += ["--eval-window-size", str(params["eval_window_size"])]

    print(f"\n[stage {stage} seed {seed}] launching:")
    print(f"  {' '.join(cmd)}")
    t0 = time.perf_counter()
    rc = subprocess.call(cmd)
    wall = time.perf_counter() - t0
    return rc, out_path, wall


def run_stage(
    ledger: dict[str, Any],
    stage: int,
    config_path: Path,
    *,
    dry_run: bool = False,
    only_seed: int | None = None,
    run_one_seed=_default_run_one_seed,
) -> None:
    """Execute (or resume) one stage, one seed at a time, with atomic ledger writes."""
    params = STAGE_PARAMS[stage]
    init_stage(ledger, stage, params)
    atomic_write_json(LEDGER_PATH, ledger)

    stage_key = str(stage)
    seeds = ledger["stages"][stage_key]["seeds"]

    if dry_run:
        pending = [s["seed"] for s in seeds if s["status"] != "COMPLETED"]
        print(f"\n[dry-run] Stage {stage} plan:")
        print(f"  params: {json.dumps(params, indent=2)}")
        print(f"  pending seeds: {pending}")
        print(f"  ledger: {LEDGER_PATH}")
        return

    for seed_entry in seeds:
        seed = seed_entry["seed"]
        if only_seed is not None and seed != only_seed:
            continue
        if seed_entry["status"] == "COMPLETED":
            print(f"[stage {stage} seed {seed}] SKIP (COMPLETED)")
            continue

        seed_entry["status"] = "RUNNING"
        seed_entry["started"] = utcnow_iso()
        atomic_write_json(LEDGER_PATH, ledger)

        try:
            rc, out_path, wall = run_one_seed(config_path, stage, seed, params)

            if rc == 0 and out_path.exists():
                try:
                    with out_path.open("r", encoding="utf-8") as f:
                        result = json.load(f)
                    seed_entry["metrics"] = extract_metrics(result)
                except Exception as e:
                    seed_entry["metrics_error"] = str(e)
                seed_entry["status"] = "COMPLETED"
                seed_entry["result_path"] = str(out_path)
            else:
                seed_entry["status"] = "FAILED"
                seed_entry["exit_code"] = rc

            seed_entry["wall_seconds"] = round(wall, 1)
            seed_entry["finished"] = utcnow_iso()
            atomic_write_json(LEDGER_PATH, ledger)

            status = seed_entry["status"]
            print(f"[stage {stage} seed {seed}] {status} in {wall:.1f}s")

            if status == "FAILED":
                print(
                    f"[stage {stage}] STOPPING on FAILED seed. Diagnose the root "
                    f"cause, then resume:\n  {resume_command(config_path, stage)}"
                )
                sys.exit(2)

        except KeyboardInterrupt:
            seed_entry["status"] = "INTERRUPTED"
            seed_entry["finished"] = utcnow_iso()
            atomic_write_json(LEDGER_PATH, ledger)
            print(f"\n[stage {stage} seed {seed}] INTERRUPTED (SIGINT).")
            print(f"Resume:\n  {resume_command(config_path, stage)}")
            sys.exit(130)


def print_stage_summary(ledger: dict[str, Any], stage: int, config_path: Path) -> None:
    seeds = ledger["stages"][str(stage)]["seeds"]
    completed = sum(1 for s in seeds if s["status"] == "COMPLETED")
    print(f"\n=== Stage {stage} summary ===")
    print(f"  {completed}/{len(seeds)} seeds COMPLETED")
    for s in seeds:
        wall = s.get("wall_seconds", "")
        wall_str = f"{wall}s" if isinstance(wall, (int, float)) else ""
        print(f"    seed {s['seed']:>2}: {s['status']:<12} {wall_str}")
    if completed < len(seeds):
        print(f"\nResume:\n  {resume_command(config_path, stage)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Gate A resumable runner")
    p.add_argument("--config", default="configs/gate_a.yaml")
    p.add_argument("--stage", type=int, choices=[1, 2], default=1)
    p.add_argument("--seed", type=int, default=None, help="Run only this seed within the stage.")
    p.add_argument("--resume", action="store_true", help="Skip seeds already COMPLETED per ledger.")
    p.add_argument("--dry-run", action="store_true", help="Validate config + print plan; no training.")
    p.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Override the stage's seed count (advanced; overrides pre-reg).",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override the stage's train_timesteps (advanced; overrides pre-reg).",
    )
    p.add_argument(
        "--evaluate",
        action="store_true",
        help="Eval-only mode is not yet implemented; use --resume to advance seeds.",
    )
    args = p.parse_args(argv)

    if args.evaluate:
        print("--evaluate is not yet wired; use --resume to advance seeds.")
        return 2

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        return 2

    # Advanced overrides — recorded in the ledger's stage params so they're
    # visible to anyone auditing the run.
    if args.num_seeds is not None:
        STAGE_PARAMS[args.stage] = {**STAGE_PARAMS[args.stage], "seeds": args.num_seeds}
    if args.steps is not None:
        STAGE_PARAMS[args.stage] = {**STAGE_PARAMS[args.stage], "train_timesteps": args.steps}

    gh = git_hash()
    ch = config_hash(config_path)

    ledger = load_ledger()
    if ledger is None:
        ledger = init_ledger(config_path, gh, ch)
    else:
        if not args.resume and args.seed is None and not args.dry_run:
            print(
                "WARNING: ledger already exists. Use --resume to continue, or "
                "delete experiments/gate_a_ledger.json to start over."
            )
            return 2
        if ledger.get("git_commit") not in (gh, "unknown") and gh != "unknown":
            print(
                f"WARNING: git hash drift. Ledger={ledger.get('git_commit', '?')[:8]} "
                f"vs HEAD={gh[:8]}. Only continue if this is expected."
            )

    print(f"experiment_id: {ledger['experiment_id']}")
    print(f"git commit:    {gh[:8]}")
    print(f"config hash:   {ch}")
    print(f"stage:         {args.stage}")
    print(f"ledger:        {LEDGER_PATH}")

    run_stage(ledger, args.stage, config_path, dry_run=args.dry_run, only_seed=args.seed)
    print_stage_summary(ledger, args.stage, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
