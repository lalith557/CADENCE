"""Offline sandbox: generate (CDAG, root-cause feature) training tuples.

The reference Part 3D specifies that the surrogate (and, by extension, the
GNN it's trained end-to-end with) is trained *offline* on synthetic drift
injections — because that's the only source of ground-truth causal labels.
This module implements that sandbox at pretrain time so Step B's GNN scorer
has labelled data to learn from before it ever sees live production traffic.

Each generated sample is:
    * `weights` — the NOTEARS output for a windowed slice under an injected drift
    * `node_features` — (n_nodes, 4) — mean/var/skew/drift_z per node per sample
    * `root_cause_feature_idx` — the injected feature (integer label)
    * `severity` — the drift magnitude (used as a regression side-target)

Sampling strategy: for each of `n_features_sampled` randomly-chosen feature
indices, draw one of the drift specs (mult/additive/tail-flip) at a random
severity, apply it to the baseline stream, build the CDAG, and record.

The resulting dataset is small enough (~O(1000) samples) to hold entirely
in memory as a list of PyG `Data` objects on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch_geometric.data import Data

from benchmarks.synthetic_drift_gen import (
    ConceptShiftSpec,
    DriftInjector,
    DriftScenario,
    DriftSchedule,
    FeatureShiftSpec,
)
from cadence.adapters.base import ModelAdapter
from cadence.attribution.gnn import cdag_to_pyg_data
from cadence.cdag import (
    ActivationTap,
    CDAGNodeSet,
    build_cdag_from_windows,
    compute_windowed_signals,
)
from cadence.common.device import get_device
from cadence.common.logging import get_logger

log = get_logger("cadence.attribution.sandbox_labels")


@dataclass
class SandboxSample:
    """One labelled (CDAG, node_features, root_cause) example."""

    data: Data  # PyG graph — already on device
    root_cause_feature_idx: int  # position within feature_indices (0..n_features-1)
    root_cause_node_idx: int  # position within node_set.specs (used for cross-entropy)
    mechanism: str  # "covariate_shift" | "concept_shift"
    severity: float


def _sample_spec(
    rng: np.random.Generator,
    feature_idx: int,
    mechanism: str,
) -> FeatureShiftSpec | ConceptShiftSpec:
    if mechanism == "covariate_shift":
        mult = float(rng.uniform(1.15, 1.6))
        add = float(rng.uniform(-1.0, 1.0))
        return FeatureShiftSpec(feature_idx=feature_idx, multiplicative=mult, additive=add)
    threshold = float(rng.uniform(-1.5, 1.5))
    flip_p = float(rng.uniform(0.15, 0.8))
    return ConceptShiftSpec(feature_idx=feature_idx, threshold=threshold, flip_probability=flip_p)


def generate_sandbox_dataset(
    *,
    adapter: ModelAdapter,
    tap: ActivationTap,
    node_set: CDAGNodeSet,
    baseline_means: np.ndarray,
    baseline_stds: np.ndarray,
    baseline_stream_X: np.ndarray,
    baseline_stream_y: np.ndarray,
    feature_names: list[str],
    n_samples: int = 400,
    window_size: int = 512,
    windows_per_sample: int = 4,
    mechanism_mix: tuple[float, float] = (0.75, 0.25),  # (covariate, concept)
    seed: int = 0,
    device: torch.device | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[SandboxSample]:
    """Roll `n_samples` synthetic drift scenarios and return labelled PyG Data.

    Each call to `adapter.predict_proba` runs on the adapter's own device (GPU
    if configured). The CDAG (NOTEARS) still runs on CPU per D-13; the GNN
    training that consumes this dataset will run on GPU.
    """
    dev = device if device is not None else get_device()
    n_feat = len(node_set.feature_indices)
    rng = np.random.default_rng(seed)
    samples: list[SandboxSample] = []
    n_use = window_size * windows_per_sample
    stream_X = baseline_stream_X[:n_use]
    stream_y = baseline_stream_y[:n_use]

    for i in range(n_samples):
        feat_idx = int(rng.integers(0, n_feat))
        mech = rng.choice(("covariate_shift", "concept_shift"), p=list(mechanism_mix))
        spec = _sample_spec(rng, feat_idx, mech)
        scenario = DriftScenario(
            name=f"sandbox_{i:04d}",
            spec=spec,
            schedule=DriftSchedule.ABRUPT,
            start_step=0,
            full_effect_step=0,
            feature_names=feature_names,
        )
        injector = DriftInjector(scenario, rng=np.random.default_rng(seed + i))
        X_drift, y_drift = injector.apply(stream_X.copy(), stream_y.copy(), step_offset=0)

        # Current F1 on the drifted window — feeds the performance-node value.
        threshold = getattr(adapter, "decision_threshold", 0.5)
        probs = adapter.predict_proba(X_drift)
        preds = (probs >= threshold).astype(np.int64)
        from sklearn.metrics import f1_score

        current_f1 = float(f1_score(y_drift, preds, zero_division=0))

        signals = compute_windowed_signals(
            X_drift,
            y_drift,
            tap,
            node_set,
            baseline_means,
            baseline_stds,
            window_f1=current_f1,
            window_size=window_size,
        )
        cdag = build_cdag_from_windows(
            signals,
            pc_alpha=0.05,
            notears_lambda=0.05,
            notears_max_iter=25,
        )
        # Node features = the last window's 4-stat vector per node.
        node_features = signals[-1]  # (n_nodes, 4)
        data = cdag_to_pyg_data(cdag.weights, node_features, device=dev)
        gt = injector.ground_truth()
        # Map feature_idx → node_idx via node_set.feature_indices.
        node_idx = node_set.feature_indices[feat_idx]
        samples.append(
            SandboxSample(
                data=data,
                root_cause_feature_idx=feat_idx,
                root_cause_node_idx=node_idx,
                mechanism=gt.mechanism,
                severity=float(gt.magnitude),
            )
        )
        if on_progress is not None and (i + 1) % max(1, n_samples // 10) == 0:
            on_progress(i + 1, n_samples)

    log.info(
        "sandbox_dataset_built",
        n=len(samples),
        n_features=n_feat,
        window_size=window_size,
    )
    return samples


# ---------- Step C: intervention-outcome triples ----------


@dataclass
class InterventionSample:
    """One (graph, candidate_node, measured post-fix F1) triple.

    The Part 3D surrogate learns MSE against `post_fix_f1`, using
    `(GNN(data)[candidate_node], pre_fix_f1)` as input.
    """

    data: Data  # PyG graph (on device) with per-node features
    candidate_feature_idx: int  # position within feature_indices (0..n_features-1)
    candidate_node_idx: int  # position within node_set.specs
    root_cause_feature_idx: int  # ground-truth root cause (for eval)
    pre_fix_f1: float  # F1 on the drifted window before any intervention
    post_fix_f1: float  # F1 on the same window after partial-retrain of candidate's layer
    severity: float
    mechanism: str


def _target_layer_for_node(node_set: CDAGNodeSet, node_idx: int, adapter: ModelAdapter) -> str:
    """Which adapter layer does this CDAG node map to?

    For feature nodes we return the first tap layer (layer1) because that's
    where raw input statistics land first — a Layer-1 partial retrain is the
    natural intervention for a "which input feature is responsible" hypothesis.
    For cluster nodes we return the layer_name recorded in the NodeSpec.
    For the performance node we default to the last tap layer.
    """
    spec = node_set.specs[node_idx]
    if spec.kind == "cluster" and spec.layer_name is not None:
        return spec.layer_name
    tap_layers = getattr(adapter, "tap_layer_names", None) or ["layer1"]
    if spec.kind == "performance":
        return tap_layers[-1]
    return tap_layers[0]


def generate_intervention_dataset(
    *,
    adapter: ModelAdapter,
    tap: ActivationTap,
    node_set: CDAGNodeSet,
    baseline_state: dict,
    baseline_threshold: float,
    baseline_means: np.ndarray,
    baseline_stds: np.ndarray,
    baseline_stream_X: np.ndarray,
    baseline_stream_y: np.ndarray,
    replay_X: np.ndarray,
    replay_y: np.ndarray,
    feature_names: list[str],
    n_drifts: int = 60,
    candidates_per_drift: int = 3,
    finetune_epochs: int = 2,
    window_size: int = 512,
    windows_per_sample: int = 3,
    mechanism_mix: tuple[float, float] = (0.75, 0.25),
    seed: int = 0,
    device: torch.device | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[InterventionSample]:
    """Generate (CDAG, candidate node, measured post-fix F1) triples.

    Procedure per drift:
      1. Inject drift with a random (feature, severity).
      2. Evaluate F1 on the drifted window -> `pre_fix_f1`.
      3. Build the CDAG once (shared across candidates for this drift).
      4. For each of `candidates_per_drift` candidate features (always
         including the true root, plus random distractors), reset the
         adapter to `baseline_state`, run a short EWC-free Layer-1
         partial retrain on the drifted window + replay buffer, and
         measure the post-fix F1 -> `post_fix_f1`.

    All F1s are computed on the same drifted window so the surrogate learns
    "delta on THIS window" and not any cross-scenario noise.
    """
    dev = device if device is not None else get_device()
    n_feat = len(node_set.feature_indices)
    rng = np.random.default_rng(seed)
    n_use = window_size * windows_per_sample
    stream_X = baseline_stream_X[:n_use]
    stream_y = baseline_stream_y[:n_use]

    replay_n = min(replay_X.shape[0], 1000)
    if replay_X.shape[0] > replay_n:
        idxs = rng.choice(replay_X.shape[0], size=replay_n, replace=False)
        replay_X_use = replay_X[idxs]
        replay_y_use = replay_y[idxs]
    else:
        replay_X_use = replay_X
        replay_y_use = replay_y

    out: list[InterventionSample] = []

    for i in range(n_drifts):
        root_feat = int(rng.integers(0, n_feat))
        mech = rng.choice(("covariate_shift", "concept_shift"), p=list(mechanism_mix))
        spec = _sample_spec(rng, root_feat, mech)
        scenario = DriftScenario(
            name=f"iv_sandbox_{i:04d}",
            spec=spec,
            schedule=DriftSchedule.ABRUPT,
            start_step=0,
            full_effect_step=0,
            feature_names=feature_names,
        )
        injector = DriftInjector(scenario, rng=np.random.default_rng(seed + i))
        X_drift, y_drift = injector.apply(stream_X.copy(), stream_y.copy(), step_offset=0)

        # Baseline (undrifted-state) F1 on the drifted window.
        adapter.load_state_dict(baseline_state)
        if hasattr(adapter, "decision_threshold"):
            adapter.decision_threshold = baseline_threshold
        threshold = getattr(adapter, "decision_threshold", 0.5)
        probs = adapter.predict_proba(X_drift)
        preds = (probs >= threshold).astype(np.int64)
        pre_fix_f1 = float(f1_score(y_drift, preds, zero_division=0))

        # Build the CDAG once (identical across candidates for this drift).
        signals = compute_windowed_signals(
            X_drift,
            y_drift,
            tap,
            node_set,
            baseline_means,
            baseline_stds,
            window_f1=pre_fix_f1,
            window_size=window_size,
        )
        cdag = build_cdag_from_windows(
            signals,
            pc_alpha=0.05,
            notears_lambda=0.05,
            notears_max_iter=25,
        )
        node_features = signals[-1]

        # Candidate features to try: always the true root + random distractors.
        distractors = rng.choice(
            [j for j in range(n_feat) if j != root_feat],
            size=max(0, candidates_per_drift - 1),
            replace=False,
        )
        cands = [root_feat] + [int(j) for j in distractors]

        for cand in cands:
            # Fresh CDAG Data on device (avoid mutating shared tensor across cands).
            data = cdag_to_pyg_data(cdag.weights, node_features, device=dev)
            node_idx = node_set.feature_indices[cand]
            target_layer = _target_layer_for_node(node_set, node_idx, adapter)

            # Reset adapter and run a short partial retrain on candidate's layer.
            adapter.load_state_dict(baseline_state)
            if hasattr(adapter, "decision_threshold"):
                adapter.decision_threshold = baseline_threshold
            try:
                adapter.partial_fit(
                    X_drift,
                    y_drift,
                    layers_to_update=[target_layer],
                    replay_X=replay_X_use,
                    replay_y=replay_y_use,
                    max_epochs=finetune_epochs,
                    finetune_lr=1e-3,
                    ewc_penalty=0.0,
                )
            except (NotImplementedError, ValueError):
                # If the target layer name doesn't exist for this adapter,
                # skip this candidate — record it as a no-op (post==pre).
                out.append(
                    InterventionSample(
                        data=data,
                        candidate_feature_idx=cand,
                        candidate_node_idx=node_idx,
                        root_cause_feature_idx=root_feat,
                        pre_fix_f1=pre_fix_f1,
                        post_fix_f1=pre_fix_f1,
                        severity=float(injector.ground_truth().magnitude),
                        mechanism=injector.ground_truth().mechanism,
                    )
                )
                continue

            probs = adapter.predict_proba(X_drift)
            preds = (probs >= adapter.decision_threshold).astype(np.int64)
            post_fix_f1 = float(f1_score(y_drift, preds, zero_division=0))

            out.append(
                InterventionSample(
                    data=data,
                    candidate_feature_idx=cand,
                    candidate_node_idx=node_idx,
                    root_cause_feature_idx=root_feat,
                    pre_fix_f1=pre_fix_f1,
                    post_fix_f1=post_fix_f1,
                    severity=float(injector.ground_truth().magnitude),
                    mechanism=injector.ground_truth().mechanism,
                )
            )

        # Reset the adapter one more time so subsequent drifts see the true baseline.
        adapter.load_state_dict(baseline_state)
        if hasattr(adapter, "decision_threshold"):
            adapter.decision_threshold = baseline_threshold

        if on_progress is not None and (i + 1) % max(1, n_drifts // 10) == 0:
            on_progress(i + 1, n_drifts)

    log.info(
        "intervention_dataset_built",
        n_drifts=n_drifts,
        n_triples=len(out),
        candidates_per_drift=candidates_per_drift,
    )
    return out
