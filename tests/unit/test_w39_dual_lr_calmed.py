"""W-39: dual-lr defaults calmed after R-Gate-A-stage1-fail-1.

The Stage-1 diagnostic run at `dual_lr=5.0, target_violation=0.0` under the
W-32-rescaled reward pushed lambda_sla to the `lambda_max=100.0` cap and
held it there for the last 30% of training (mean=100.00 over 9,216 dual
updates in seed 0's log). Under saturated lambda, PPO learned a near-
always-retrain policy, which:
  * FAILED gate G6 (mean lambda last-30% must be <= 50).
  * FAILED gate G7 (cross-seed mean_f1 std must be >= 0.02; deterministic
    always-retrain gave std = 0.0021).
  * Contributed to G8 (wall > 4h) via excess retraining.

Single-fix per pre-registration Part 1 loop: reduce dual_lr default from
5.0 to 1.0. This test pins the defaults so they can't silently regress.

Related weakness: W-39 in docs/weaknesses.md; result: R-Gate-A-stage1-fail-1
in docs/results.md.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _default_of(source: str, class_name: str, field_name: str) -> object:
    """Extract a default from a `@dataclass class X: field: T = <default>` block."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    if stmt.target.id == field_name and stmt.value is not None:
                        return ast.literal_eval(stmt.value)
    raise AssertionError(f"{class_name}.{field_name} not found in source")


def test_augmented_lagrangian_default_dual_lr_is_calmed() -> None:
    """AugmentedLagrangianConfig.dual_lr default must be 1.0 (was 5.0).

    Stage-1 fail evidence: with dual_lr=5.0, lambda climbed to 100.0 and
    stayed there for the whole last 30% of training, violating G6."""
    src = (REPO / "cadence" / "rso" / "lagrangian.py").read_text(encoding="utf-8")
    dual_lr = _default_of(src, "AugmentedLagrangianConfig", "dual_lr")
    assert dual_lr == 1.0, (
        f"AugmentedLagrangianConfig.dual_lr default = {dual_lr}, expected 1.0. "
        f"Reverting from the W-39 calming is a regression that will re-trip "
        f"pre-reg G6/G7. See R-Gate-A-stage1-fail-1 in docs/results.md."
    )


def test_augmented_lagrangian_bounds_unchanged() -> None:
    """lambda bounds + ema_beta unchanged by W-39 (dual_lr only).

    NOTE: `target_violation` was 0.0 at W-39 time but was later raised to 0.13
    by W-41 (R-Gate-A-stage1-fail-2) — it is now owned and pinned by
    tests/unit/test_w41_target_violation.py, not here. Each was a separate
    single-knob loop iteration.
    """
    src = (REPO / "cadence" / "rso" / "lagrangian.py").read_text(encoding="utf-8")
    assert _default_of(src, "AugmentedLagrangianConfig", "lambda_init") == 1.0
    assert _default_of(src, "AugmentedLagrangianConfig", "lambda_min") == 0.0
    assert _default_of(src, "AugmentedLagrangianConfig", "lambda_max") == 100.0
    assert _default_of(src, "AugmentedLagrangianConfig", "ema_beta") == 0.5


def test_phase_a_run_dual_lr_arg_default_is_calmed() -> None:
    """`benchmarks/phase_a_run.py --dual-lr` default must be 1.0 to match.

    If the CLI default disagrees with the AugmentedLagrangianConfig default,
    Stage-1 runs launched via `run_experiment.py` (which does not pass
    `--dual-lr`) would silently get the CLI default not the dataclass
    default — the sort of drift the pre-reg protocol is designed to catch.
    """
    src = (REPO / "benchmarks" / "phase_a_run.py").read_text(encoding="utf-8")
    # Look for `p.add_argument("--dual-lr", type=float, default=<v>)`.
    tree = ast.parse(src)
    found_defaults: list[float] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_argument":
                # Match by first positional arg.
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--dual-lr":
                    for kw in node.keywords:
                        if kw.arg == "default":
                            found_defaults.append(ast.literal_eval(kw.value))
    assert found_defaults, "`--dual-lr` add_argument not found in phase_a_run.py"
    assert found_defaults[0] == 1.0, (
        f"phase_a_run --dual-lr default = {found_defaults[0]}, expected 1.0 "
        f"(W-39 fix). See R-Gate-A-stage1-fail-1."
    )
