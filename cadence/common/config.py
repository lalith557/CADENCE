"""Typed config loader — YAML → pydantic models.

Two profiles live in `configs/`:
    * `default.yaml` — the config the paper reports numbers under; sized for
      the reference-spec GTX 1650 (4 GB VRAM) per D-11.
    * `local.yaml` — allowed to use the dev machine's extra headroom; must NOT
      be used for anything that lands in `docs/results.md`.

Load with:
    from cadence.common.config import load_config
    cfg = load_config("configs/default.yaml")
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    root: Path = Field(default=Path("Dataset"))
    fraud_csv: Path = Field(default=Path("Dataset/Credit Card Fraud Detection/creditcard.csv"))
    gmsc_train_csv: Path = Field(default=Path("Dataset/Give Me Some Credit/cs-training.csv"))
    telco_csv: Path = Field(
        default=Path("Dataset/Telco Customer Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    )
    adult_dir: Path = Field(default=Path("Dataset/adult"))
    torchvision_cache: Path = Field(default=Path(".cadence_cache/torchvision"))
    test_size: float = 0.2
    val_size: float = 0.1


class ModelConfig(BaseModel):
    """FraudNet defaults per D-1."""

    hidden_dims: list[int] = Field(default_factory=lambda: [64, 32])
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 30
    early_stopping_patience: int = 5
    class_weighted: bool = True


class CDAGConfig(BaseModel):
    """CDAG defaults per D-2, D-3."""

    layer_1_clusters: int = 8
    layer_2_clusters: int = 4
    cluster_k_sweep: tuple[int, int] = (2, 16)
    window_size: int = 2048
    stats: list[str] = Field(default_factory=lambda: ["mean", "var", "skew"])
    pc_alpha: float = 0.05
    notears_lambda: float = 0.01
    notears_max_iter: int = 100


class RSOConfig(BaseModel):
    """PPO reward weights per D-7. Sensitivity sweep is W-7."""

    top_k: int = 5
    w_gpu_hr: float = 0.05
    w_kg_co2: float = 0.5
    lambda_sla_init: float = 1.0
    lambda_update_period: int = 10
    ppo_lr: float = 3e-4
    ppo_clip: float = 0.2
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95


class ExecutorConfig(BaseModel):
    """EWC defaults per D-8, multi-node fallback per D-12."""

    fisher_sample_size: int = 5000
    replay_ratio_new: float = 0.7
    finetune_lr: float = 1e-4
    finetune_max_epochs: int = 10
    ewc_lambda: float = 1000.0
    tau_partial_unfreeze: float = 0.15
    tau_diffuse_escalate: float = 0.60


class SLAConfig(BaseModel):
    """SLA = 0.95 × baseline_F1 per W-6 (default; per-dataset override allowed)."""

    f1_relative_floor: float = 0.95
    shadow_window_hours: int = 24
    forgetting_epsilon: float = 0.02


class TrainingBudgetConfig(BaseModel):
    """Determines what heavy operations are allowed."""

    max_vram_gb: float = 4.0  # paper number; local.yaml raises this
    seeds_headline: int = 10  # W-5
    seeds_ablation: int = 5


class ExperimentConfig(BaseModel):
    seed: int = 42
    mlflow_uri: str = "file:./experiments/mlruns"
    experiment_name: str = "cadence-default"


class Config(BaseModel):
    profile: Literal["default", "local", "ci"] = "default"
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    cdag: CDAGConfig = CDAGConfig()
    rso: RSOConfig = RSOConfig()
    executor: ExecutorConfig = ExecutorConfig()
    sla: SLAConfig = SLAConfig()
    budget: TrainingBudgetConfig = TrainingBudgetConfig()
    experiment: ExperimentConfig = ExperimentConfig()

    @model_validator(mode="after")
    def _enforce_reproducibility_invariant(self) -> Config:
        # Paper numbers must come from `default` or `ci` profile.
        # A `local` config is allowed a bigger VRAM budget for iteration only.
        if self.profile == "default" and self.budget.max_vram_gb > 4.0:
            raise ValueError(
                "default profile must fit within 4 GB VRAM (paper reproducibility); "
                "raise VRAM budget only in the `local` profile."
            )
        return self

    def hash(self) -> str:
        """Stable content hash for MLflow logging."""
        blob = self.model_dump_json(exclude={"experiment"}).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return Config.model_validate(raw)


def dump_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            json.loads(cfg.model_dump_json()),
            f,
            sort_keys=False,
            default_flow_style=False,
        )
