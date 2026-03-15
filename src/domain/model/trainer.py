"""
Trainer Protocol / FoldResult / TrainResult — モデル学習ドメイン。

設計方針:
  - Trainer は Protocol。LightGBMTrainer / PyTorchTrainer など実装は
    infrastructure 層に置き、usecase は Protocol にのみ依存する。
  - FoldResult は 1 fold 分の学習結果を保持する。
  - TrainResult は全 fold を集約した最終結果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class FoldResult:
    """1 fold 分の学習結果。"""

    fold_idx: int
    train_score: float
    valid_score: float
    metric: str
    model_path: Path
    oof_path: Path
    error_analysis_path: Path
    feature_importance_path: Path | None
    n_train: int
    n_valid: int
    best_iteration: int | None = None
    feature_importances: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainResult:
    """全 fold を集約した学習結果。"""

    job_id: str
    timestamp: str
    commit_hash: str
    trainer_type: str
    output_dir: Path
    fold_results: list[FoldResult]
    cv_mean_score: float
    cv_std_score: float
    metric: str
    seed: int
    trainer_fallback: bool = False
    trainer_requested: str | None = None


@runtime_checkable
class Trainer(Protocol):
    """モデル学習の抽象 Protocol。

    LightGBMTrainer・PyTorchTrainer など具体実装は
    infrastructure/trainer/ に置く。
    """

    def fit_folds(
        self,
        preprocess_output_dir: Path,
        output_dir: Path,
        cfg: dict[str, Any],
    ) -> TrainResult: ...
