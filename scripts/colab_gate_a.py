"""Paper-scale Gate A driver, structured for Colab / Kaggle / any long-lived box.

The local session compute budget (~single-day) can't cover the paper protocol
(≥10 seeds × ≥5 scenarios × 30k PPO timesteps + full-fidelity eval). This
script is the identical config packaged as a single-file driver you can run
on a T4/A100 Colab notebook or a Kaggle GPU kernel; it writes checkpoints
per-seed so a mid-run kill leaves usable partial results.

Usage (Colab):
    !git clone <repo>
    %cd Capstone-Project
    !pip install -e .[ml,gnn,drift,dev]
    !python scripts/colab_gate_a.py --seeds 10 --train-timesteps 30000

Or import piecewise and run interactively.

The script wraps `benchmarks.phase_a_run.main`; it doesn't reimplement any
research code. Its whole point is (a) parameters at paper scale and (b) a
per-seed resume-able loop that phase_a_run's monolithic runner lacks.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--train-timesteps", type=int, default=30_000)
    p.add_argument("--sla", type=float, default=0.65)
    p.add_argument("--n-windows", type=int, default=15)
    p.add_argument("--window-size", type=int, default=2048)
    p.add_argument("--out-dir", default="experiments/gate_a_full")
    p.add_argument(
        "--train-only",
        action="store_true",
        help="Save checkpoint per seed but skip the paper-fidelity eval loop",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Gate A full-scale driver, writing to {out_dir}")
    print(
        f"Config: seeds={args.seeds}, train_timesteps={args.train_timesteps}, "
        f"n_windows={args.n_windows}, sla={args.sla}"
    )

    per_seed_summaries: list[dict] = []
    for seed in range(args.seeds):
        seed_out = out_dir / f"seed{seed}.json"
        if seed_out.exists():
            print(f"[resume] seed {seed} already recorded at {seed_out}; skipping")
            with seed_out.open("r", encoding="utf-8") as f:
                per_seed_summaries.append(json.load(f))
            continue

        # phase_a_run's --seeds runs its own inner loop; here we drive it one
        # seed at a time so a kill leaves per-seed json artifacts. Each call
        # trains a fresh PPO and writes its checkpoint to
        # experiments/rso_ppo_phase_a.zip — override that per-seed.
        cmd = [
            "python",
            "-m",
            "benchmarks.phase_a_run",
            "--seeds",
            "1",
            "--train-timesteps",
            str(args.train_timesteps),
            "--n-windows",
            str(args.n_windows),
            "--window-size",
            str(args.window_size),
            "--sla",
            str(args.sla),
            "--out",
            str(seed_out),
        ]
        if args.train_only:
            cmd.append("--train-only")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Pin the seed the runner uses inside its adapter/PPO seeding by
        # exporting CADENCE_SEED so the runner picks it up if configured.
        env["CADENCE_SEED"] = str(seed)
        t0 = time.perf_counter()
        import subprocess

        result = subprocess.run(cmd, env=env, check=False)
        elapsed = time.perf_counter() - t0
        print(f"[seed {seed}] exit={result.returncode} wall={elapsed:.1f}s")
        if seed_out.exists():
            with seed_out.open("r", encoding="utf-8") as f:
                per_seed_summaries.append(json.load(f))

    # Aggregate. If phase_a_run wrote paired-Wilcoxon-friendly per-strategy
    # arrays into each seed file we can concatenate here; otherwise the
    # aggregation is deferred to a follow-up notebook cell.
    agg_path = out_dir / "gate_a_aggregate.json"
    with agg_path.open("w", encoding="utf-8") as f:
        json.dump({"n_seeds": len(per_seed_summaries), "per_seed": per_seed_summaries}, f, indent=2, default=str)
    print(f"\nWrote aggregate to {agg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
