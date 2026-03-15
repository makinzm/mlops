"""
LightGBM Trainer — Trainer Protocol の実装。

Phase 3 で本実装を行う。現時点はスタブ（fit_folds シグネチャのみ）。
"""

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.domain.model.trainer import TrainResult


class LightGBMTrainer:
    """LightGBM を使った k-fold クロスバリデーション学習器。"""

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    def fit_folds(
        self,
        preprocess_output_dir: Path,
        output_dir: Path,
        cfg: dict[str, Any],
    ) -> TrainResult:
        raise NotImplementedError("Phase 3 で実装予定")
