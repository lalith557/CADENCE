"""Phase F, part 4 — text vocabulary drift on Yelp reviews (W-29 fix).

Pretrain a TF-IDF + LR sentiment classifier on early Yelp reviews
(<= 2013). Stream the late reviews (>= 2018) in windows and grade the
same 3 strategies as the tabular / real-drift runners:

  * cadence_rule    — RetrainExecutor + EscalationLadder + FalseAlarmConfig
  * periodic        — full retrain every N windows
  * reactive_full   — PSI on TF-IDF summary stats -> full retrain

Because a text adapter has no layer-level partial retrain, the executor
falls back to full retrain per §4 contract. Cadence's benefit here is
the false-alarm suppression + escalation ladder + shadow validation,
not layer-targeted intervention — a scoped honesty about what tabular
CADENCE gives you in the text setting.

Usage:
    python -m benchmarks.phase_f_text --seeds 2 --n-windows 8 --per-slice-cap 8000
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import f1_score

from cadence.adapters.text import TextConfig, TFIDFLogRegAdapter
from cadence.common.config import load_config
from cadence.common.logging import get_logger
from cadence.common.seeds import set_global_seed
from cadence.common.tracking import start_run
from cadence.data.text_yelp import load_yelp_domain_shift, load_yelp_slices
from cadence.executor import (
    EscalationConfig,
    EscalationDecision,
    EscalationLadder,
    ExecutorConfig,
    RetrainExecutor,
)
from cadence.executor.fallbacks import FalseAlarmConfig, is_false_alarm

log = get_logger("cadence.benchmarks.phase_f_text")


def _mean_std(vs):
    vs = list(vs)
    if not vs:
        return (0.0, 0.0)
    return (
        float(statistics.fmean(vs)),
        float(statistics.pstdev(vs)) if len(vs) > 1 else 0.0,
    )


def _f1(adapter, X, y) -> float:
    if len(X) == 0:
        return 0.0
    probs = adapter.predict_proba(X)
    preds = (probs >= adapter.decision_threshold).astype(np.int64)
    return float(f1_score(y, preds, zero_division=0))


def _psi_on_tfidf(baseline_matrix: np.ndarray, live_matrix: np.ndarray, top_k: int = 200) -> float:
    """Very cheap drift signal on TF-IDF: PSI over the top-`top_k`
    baseline vocabulary terms' mean rate. Not a rigorous PSI trigger,
    but a proportional analogue that fires when the vocabulary shifts.
    """
    b_mean = np.asarray(baseline_matrix.mean(axis=0)).flatten()
    l_mean = np.asarray(live_matrix.mean(axis=0)).flatten()
    order = np.argsort(-b_mean)[:top_k]
    b = b_mean[order] + 1e-6
    l_ = l_mean[order] + 1e-6
    b = b / b.sum()
    l_ = l_ / l_.sum()
    return float(np.sum((l_ - b) * np.log(l_ / b)))


def _windowed_run(
    *,
    adapter: TFIDFLogRegAdapter,
    baseline_state: dict,
    stream_texts: list[str],
    stream_y: np.ndarray,
    stream_baseline_matrix: np.ndarray,
    trigger_matrix: np.ndarray,
    strategy_name: str,
    sla_target: float,
    window_size: int,
    n_windows: int,
    executor_cfg: ExecutorConfig,
    historical_texts: list[str],
    historical_y: np.ndarray,
    periodic_period: int = 4,
    psi_threshold: float = 0.25,
) -> dict:
    adapter.load_state_dict(baseline_state)
    exec_ = RetrainExecutor(
        adapter,
        historical_X=np.array(historical_texts, dtype=object),
        historical_y=historical_y,
        cfg=executor_cfg,
    )
    ladder = EscalationLadder(EscalationConfig(ladder=("partial", "full"), unrecoverable_streak=3))

    rolling_f1: list[float] = []
    action_counts = {"no-op": 0, "partial": 0, "full": 0}
    n_rollbacks = 0
    n_escalated = 0
    n_periodic_since = 0

    for w in range(n_windows):
        s = w * window_size
        e = min(s + window_size, len(stream_texts))
        if e <= s:
            break
        window_texts = stream_texts[s:e]
        window_y = stream_y[s:e]
        pre_f1 = _f1(adapter, window_texts, window_y)
        rolling_f1.append(pre_f1)

        window_texts_arr = np.array(window_texts, dtype=object)

        if strategy_name == "cadence_rule":
            # Cheap TF-IDF PSI on this window vs baseline train set.
            live_matrix = adapter.vectorizer.transform(window_texts)
            psi = _psi_on_tfidf(trigger_matrix, live_matrix, top_k=200)
            psi_fired = psi >= psi_threshold
            if is_false_alarm(psi_fired, pre_f1, cfg=FalseAlarmConfig(sla_target=sla_target)):
                action_counts["no-op"] += 1
                continue
            if pre_f1 >= sla_target:
                action_counts["no-op"] += 1
                continue
            # Text adapter has no tap layers → partial NIE in the ladder →
            # ladder either catches or we fall through to full.
            try:
                result = ladder.run_ladder(
                    exec_,
                    window_X=window_texts_arr,
                    window_y=window_y,
                    target_layer_for_partial="text_stub",
                )
            except NotImplementedError:
                report = exec_.execute(
                    action="full", window_X=window_texts_arr, window_y=window_y
                )
                action_counts["full"] += 1
                if report.event.was_rolled_back:
                    n_rollbacks += 1
                continue

            for report in result.reports:
                action_counts[report.action_taken] += 1
                if report.event.was_rolled_back:
                    n_rollbacks += 1
            if result.decision == EscalationDecision.ESCALATED_HUMAN:
                n_escalated += 1

        elif strategy_name == "periodic":
            n_periodic_since += 1
            if n_periodic_since >= periodic_period:
                report = exec_.execute(
                    action="full", window_X=window_texts_arr, window_y=window_y
                )
                action_counts["full"] += 1
                if report.event.was_rolled_back:
                    n_rollbacks += 1
                n_periodic_since = 0
            else:
                action_counts["no-op"] += 1

        elif strategy_name == "reactive_full":
            live_matrix = adapter.vectorizer.transform(window_texts)
            psi = _psi_on_tfidf(trigger_matrix, live_matrix, top_k=200)
            if psi >= psi_threshold:
                report = exec_.execute(
                    action="full", window_X=window_texts_arr, window_y=window_y
                )
                action_counts["full"] += 1
                if report.event.was_rolled_back:
                    n_rollbacks += 1
            else:
                action_counts["no-op"] += 1
        else:
            raise ValueError(strategy_name)

    return {
        "strategy": strategy_name,
        "mean_f1": float(np.mean(rolling_f1)) if rolling_f1 else 0.0,
        "min_f1": float(np.min(rolling_f1)) if rolling_f1 else 0.0,
        "action_counts": action_counts,
        "n_rollbacks": n_rollbacks,
        "n_escalated_human": n_escalated,
        "rolling_f1": rolling_f1,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--per-slice-cap", type=int, default=8_000)
    p.add_argument("--max-lines-scan", type=int, default=300_000)
    p.add_argument(
        "--split",
        choices=["temporal", "domain"],
        default="temporal",
        help="temporal = early (<=2013) vs late (>=2018); domain = extreme (1/5★) vs marginal (2/4★)",
    )
    p.add_argument("--window-size", type=int, default=800)
    p.add_argument("--n-windows", type=int, default=8)
    p.add_argument("--sla", type=float, default=0.85)
    p.add_argument("--periodic-period", type=int, default=4)
    p.add_argument("--psi-threshold", type=float, default=0.15)
    p.add_argument("--out", default="experiments/phase_f_text.json")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    log.info(
        "phase_f_text_start", seeds=args.seeds, per_slice_cap=args.per_slice_cap
    )

    if args.split == "domain":
        early, late = load_yelp_domain_shift(
            per_slice_cap=args.per_slice_cap, max_lines_scan=args.max_lines_scan
        )
    else:
        early, late = load_yelp_slices(
            per_slice_cap=args.per_slice_cap, max_lines_scan=args.max_lines_scan
        )
    log.info("yelp_early_or_extreme", **early.summary())
    log.info("yelp_late_or_marginal", **late.summary())

    # Pretrain once (seed=42): train on early slice.
    set_global_seed(42)
    adapter = TFIDFLogRegAdapter(TextConfig())
    n_val = max(500, int(0.10 * len(early.texts)))
    adapter.fit(
        early.texts[:-n_val],
        early.y[:-n_val],
        X_val=early.texts[-n_val:],
        y_val=early.y[-n_val:],
    )
    baseline_state = adapter.state_dict()
    baseline_f1 = _f1(adapter, early.texts[-n_val:], early.y[-n_val:])

    # Undrifted stream F1 = baseline (early) performance on the LATE slice.
    undrifted_stream_f1 = _f1(
        adapter,
        late.texts[: args.window_size * args.n_windows],
        late.y[: args.window_size * args.n_windows],
    )
    log.info(
        "baseline_ready",
        baseline_f1=baseline_f1,
        undrifted_stream_f1=undrifted_stream_f1,
        sla_target=args.sla,
        vocab_size=len(adapter.vectorizer.vocabulary_),
    )

    # Precompute the trigger reference matrix once.
    trigger_matrix = adapter.vectorizer.transform(
        early.texts[: min(3000, len(early.texts))]
    )

    strategies = ("cadence_rule", "periodic", "reactive_full")
    results: dict[str, list[dict]] = {s: [] for s in strategies}
    exec_cfg = ExecutorConfig(
        sla_target=args.sla,
        finetune_epochs=1,
        fullretrain_epochs=1,
    )

    with start_run(cfg, run_name="phase_f_text_yelp"):
        mlflow.log_param("phase", "F/text")
        mlflow.log_param("adapter_family", adapter.family)
        mlflow.log_param("dataset", "yelp_reviews")
        mlflow.log_param("seeds", args.seeds)
        mlflow.log_param("per_slice_cap", args.per_slice_cap)
        mlflow.log_param("window_size", args.window_size)
        mlflow.log_param("n_windows", args.n_windows)
        mlflow.log_metric("baseline_train_f1", baseline_f1)
        mlflow.log_metric("undrifted_stream_f1", undrifted_stream_f1)
        mlflow.log_metric("vocab_size", len(adapter.vectorizer.vocabulary_))

        for seed in range(args.seeds):
            set_global_seed(seed)
            for strat_name in strategies:
                t0 = time.perf_counter()
                out = _windowed_run(
                    adapter=adapter,
                    baseline_state=baseline_state,
                    stream_texts=late.texts,
                    stream_y=late.y,
                    stream_baseline_matrix=trigger_matrix,
                    trigger_matrix=trigger_matrix,
                    strategy_name=strat_name,
                    sla_target=args.sla,
                    window_size=args.window_size,
                    n_windows=args.n_windows,
                    executor_cfg=exec_cfg,
                    historical_texts=early.texts,
                    historical_y=early.y,
                    periodic_period=args.periodic_period,
                    psi_threshold=args.psi_threshold,
                )
                out["wall_s"] = time.perf_counter() - t0
                out["seed"] = seed
                results[strat_name].append(out)
                log.info(
                    "strategy_done",
                    strategy=strat_name,
                    seed=seed,
                    mean_f1=out["mean_f1"],
                    actions=out["action_counts"],
                    rollbacks=out["n_rollbacks"],
                )
                gc.collect()

        def _agg(name):
            r = results[name]
            return {
                "n_seeds": len(r),
                "mean_f1": _mean_std(x["mean_f1"] for x in r),
                "min_f1": _mean_std(x["min_f1"] for x in r),
                "actions_full": _mean_std(x["action_counts"]["full"] for x in r)[0],
                "actions_partial": _mean_std(x["action_counts"]["partial"] for x in r)[0],
                "actions_noop": _mean_std(x["action_counts"]["no-op"] for x in r)[0],
                "rollbacks_mean": _mean_std(x["n_rollbacks"] for x in r)[0],
            }

        summary = {
            "adapter_family": adapter.family,
            "early_summary": early.summary(),
            "late_summary": late.summary(),
            "baseline_train_f1": baseline_f1,
            "undrifted_stream_f1": undrifted_stream_f1,
            "vocab_size": len(adapter.vectorizer.vocabulary_),
            "aggregates": {s: _agg(s) for s in strategies},
            "config": vars(args),
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        mlflow.log_artifact(args.out)

        print("\n=== Gate F (text): Yelp early -> late vocabulary drift ===")
        print(
            f"Baseline train F1: {baseline_f1:.4f}  Undrifted stream F1: {undrifted_stream_f1:.4f}  "
            f"SLA: {args.sla}  Vocab: {len(adapter.vectorizer.vocabulary_)}"
        )
        print(
            f"{'strategy':<18}{'mean_f1':>18}{'min_f1':>18}{'no-op':>10}{'partial':>10}{'full':>10}{'rollbacks':>12}"
        )
        for name, a in summary["aggregates"].items():
            print(
                f"{name:<18}"
                f"{a['mean_f1'][0]:>10.4f} +/- {a['mean_f1'][1]:.4f}"
                f"{a['min_f1'][0]:>10.4f} +/- {a['min_f1'][1]:.4f}"
                f"{a['actions_noop']:>10.1f}"
                f"{a['actions_partial']:>10.1f}"
                f"{a['actions_full']:>10.1f}"
                f"{a['rollbacks_mean']:>12.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
