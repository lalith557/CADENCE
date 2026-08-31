"""CADENCE CLI — `cadence <command>`.

Phase 0 exposes only the introspection commands. Real pipeline commands
(`train-baseline`, `inject-drift`, `run-pipeline`) land in later phases.
"""

from __future__ import annotations

from pathlib import Path

import click

from cadence import __version__
from cadence.common.config import load_config
from cadence.common.logging import get_logger


@click.group()
@click.version_option(__version__)
def main() -> None:
    """CADENCE — Causal Attribution-Driven Efficient Continual Retraining."""


@main.command("config-show")
@click.option("--config", "-c", "config_path", default="configs/default.yaml", show_default=True)
def config_show(config_path: str) -> None:
    """Print the resolved config + its hash — used by MLflow for reproducibility."""
    cfg = load_config(config_path)
    click.echo(cfg.model_dump_json(indent=2))
    click.echo(f"\nconfig_hash: {cfg.hash()}")


@main.command("datasets-check")
@click.option("--config", "-c", "config_path", default="configs/default.yaml", show_default=True)
def datasets_check(config_path: str) -> None:
    """Verify each required dataset file is on disk."""
    from cadence.data.loaders import REGISTRY, DatasetNotFoundError

    log = get_logger("cadence.cli")
    cfg = load_config(config_path)
    ok = True
    for name, loader in REGISTRY.items():
        try:
            ds = loader(cfg.data, seed=cfg.experiment.seed)
            log.info("dataset_ok", **ds.summary())
        except DatasetNotFoundError as e:
            ok = False
            log.warning(
                "dataset_missing",
                dataset=name,
                path=str(e.path),
                download_url=e.download_url,
            )
    if not ok:
        raise SystemExit(1)


@main.command("gpu-check")
@click.option(
    "--matmul-seconds",
    default=1.0,
    show_default=True,
    help="Wall-clock seconds to hammer a cuda matmul, proving real execution.",
)
def gpu_check(matmul_seconds: float) -> None:
    """Prove CUDA is really being used.

    Per Execution Plan §5: prints torch build, cuda availability, device
    name, free/total VRAM, runs a real matmul on the card, and reports
    peak VRAM. Exits non-zero if a claimed cuda device silently degrades.
    """
    import time

    import torch

    from cadence.common.device import cuda_memory_snapshot, describe, reset_peak_memory

    click.echo(f"torch:            {torch.__version__}")
    click.echo(f"torch.version.cuda: {torch.version.cuda}")
    click.echo(f"cuda available:   {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        click.echo("no cuda device — gpu-check FAILED (Execution Plan §1b)")
        raise SystemExit(2)

    info = describe(torch.device("cuda"))
    click.echo(f"device:           {info.name}")
    click.echo(f"vram total (GB):  {info.total_vram_bytes / 1024**3:.3f}")
    click.echo(f"vram free (GB):   {info.free_vram_bytes / 1024**3:.3f}")

    reset_peak_memory()
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 0
    while time.perf_counter() - t0 < matmul_seconds:
        c = a @ b
        iters += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    snap = cuda_memory_snapshot()
    peak_mb = snap["max_allocated"] / 1024**2
    tflops_measured = (2 * 4096**3 * iters) / dt / 1e12

    click.echo(f"matmul iters:     {iters} in {dt:.3f}s")
    click.echo(f"measured tflops:  {tflops_measured:.2f}")
    click.echo(f"peak vram (MB):   {peak_mb:.2f}")
    if snap["max_allocated"] <= 0:
        click.echo("zero VRAM allocated — matmul did not run on cuda")
        raise SystemExit(3)
    # Fingerprint c so the compiler can't elide the loop.
    _ = float(c.sum().detach().cpu().item())
    click.echo("gpu-check OK")


@main.command("smoke")
@click.option("--config", "-c", "config_path", default="configs/ci.yaml", show_default=True)
def smoke(config_path: str) -> None:
    """Fast end-to-end smoke: load config, set seeds, log to MLflow, exit."""
    from cadence.common.seeds import set_global_seed
    from cadence.common.tracking import start_run

    log = get_logger("cadence.cli")
    cfg = load_config(config_path)
    report = set_global_seed(cfg.experiment.seed)
    log.info("seed_set", **report.__dict__)

    Path("experiments").mkdir(exist_ok=True)
    with start_run(cfg, run_name="phase0-smoke") as run:
        log.info("smoke_run_started", run_id=run.info.run_id)
    log.info("smoke_run_ok")


@main.command("api")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option(
    "--config",
    "-c",
    "config_path",
    default="configs/default.yaml",
    show_default=True,
    help="Config used by bootstrap_from_config to seed the API's adapter + trigger + policy.",
)
@click.option(
    "--no-bootstrap",
    is_flag=True,
    help="Skip bootstrap_from_config; start with a bare API (client must POST /admin/reload).",
)
def api(host: str, port: int, config_path: str, no_bootstrap: bool) -> None:
    """Run the FastAPI service (cadence.api.server) via uvicorn."""
    import uvicorn

    from cadence.api.server import app, bootstrap_from_config

    log = get_logger("cadence.cli")
    if not no_bootstrap:
        log.info("api_bootstrap_start", config=config_path)
        bootstrap_from_config(config_path)
        log.info("api_bootstrap_done")
    uvicorn.run(app, host=host, port=port, log_level="info")


@main.command("dashboard")
@click.option("--port", default=8501, show_default=True, type=int)
@click.option("--host", default="127.0.0.1", show_default=True)
def dashboard(port: int, host: str) -> None:
    """Launch the Streamlit dashboard (dashboard/app.py)."""
    import subprocess
    import sys

    dash_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    if not dash_path.exists():
        raise click.ClickException(f"dashboard/app.py not found at {dash_path}")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dash_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    click.echo(f"launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
