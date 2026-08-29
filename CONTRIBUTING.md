# Contributing to CADENCE

This project has two simultaneous bars: **research-grade** (paper-worthy) and **product-grade** (launchable). Please read them both before opening a PR.

## Non-negotiables

1. **No fabricated numbers.** Every metric that lands in `docs/results.md` must come from a real run, with a config hash + MLflow run ID + seed. If a hypothesis fails, report the failure — negative results are valuable.
2. **Update the living docs in the same commit as the code.** If you make a design choice, add a `D-<n>` entry to `docs/decisions.md`. If you find a weakness, add a `W-<n>` entry to `docs/weaknesses.md`. Do not batch these to the end.
3. **Reproducibility.** Fixed seeds, deterministic mode where possible, all configs tracked in MLflow. Use `configs/default.yaml` for anything that lands in the paper.
4. **Small PRs.** One decision or one weakness fix per PR is ideal.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .[dev]
make test
```

## Style

- `make lint` / `make format` must pass (ruff).
- `make typecheck` (mypy) is advisory during Phase 0, enforced later.
- Follow `docs/decisions.md` conventions — any deviation from a decision must add a new decision entry.

## Reporting weaknesses

Open an issue titled `[weakness] <title>` and populate the `docs/weaknesses.md` template. Blockers/majors get fixed before the next phase gate; minors get logged.

## Reproducing an experiment

Every entry in `docs/results.md` includes a `Command to reproduce`. Run it with the same MLflow tracking URI to compare.
