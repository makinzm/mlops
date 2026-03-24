"""
Phase 3: VisionTrainer の統合テスト。

なぜこのテストが必要か:
  - VisionTrainer は Trainer Protocol を満たし、fit_folds() が
    fold ごとに画像分類モデルを学習して TrainResult を返すことを確認する。
  - 合成画像（8x8 PNG）を使って実際に学習が動くことを確認する。
  - output は input と異なる Dir であること（再現性保証）。
  - seed で再現性が保たれること。
  - model.pt, oof_train.parquet, error_analysis.parquet が保存されること。

時間計算量: O(E * N * B) — E: エポック数, N: サンプル数, B: バッチ処理
空間計算量: O(P + N) — P: モデルパラメータ数, N: サンプル数
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from PIL import Image

from src.domain.model.trainer import TrainResult
from src.infrastructure.trainer.vision_trainer import VisionTrainer


def _create_synthetic_images(
    image_dir: Path, num_images: int, size: int = 8, seed: int = 42
) -> list[str]:
    """テスト用の合成画像を生成する。

    時間計算量: O(num_images * size^2)
    空間計算量: O(size^2)
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths: list[str] = []
    for i in range(num_images):
        data = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        img = Image.fromarray(data)
        path = image_dir / f"image_{i:04d}.png"
        img.save(path)
        paths.append(str(path))
    return paths


@pytest.fixture
def vision_fold_dir(tmp_path: Path) -> Path:
    """fold_0/ 配下に train.parquet / test.parquet + 合成画像を持つ前処理出力ディレクトリ。"""
    fold0 = tmp_path / "preprocess_out" / "fold_0"
    fold0.mkdir(parents=True)

    image_dir = tmp_path / "images"
    rng = np.random.default_rng(42)
    n_train, n_valid = 20, 10

    train_paths = _create_synthetic_images(image_dir / "train", n_train, seed=42)
    valid_paths = _create_synthetic_images(image_dir / "valid", n_valid, seed=99)

    train_df = pl.DataFrame(
        {
            "image_path": train_paths,
            "label": rng.integers(0, 2, n_train).tolist(),
        }
    )
    valid_df = pl.DataFrame(
        {
            "image_path": valid_paths,
            "label": rng.integers(0, 2, n_valid).tolist(),
        }
    )

    train_df.write_parquet(fold0 / "train.parquet")
    valid_df.write_parquet(fold0 / "test.parquet")
    return tmp_path / "preprocess_out"


@pytest.fixture
def vision_cfg() -> dict[str, Any]:
    return {
        "job_id": "test_vision",
        "target_col": "label",
        "image_path_col": "image_path",
        "trainer": {"type": "vision"},
        "backbone": {
            "name": "simple_cnn",
            "pretrained": False,
            "image_size": 32,
        },
        "training": {
            "num_epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "num_workers": 0,
        },
        "num_classes": 2,
        "report": {"n_error_samples": 3},
        "seed": 42,
    }


_FAKE_COMMIT = "c" * 40


class TestVisionTrainerFitFolds:
    def _run(self, vision_fold_dir: Path, tmp_path: Path, cfg_dict: dict[str, Any]) -> TrainResult:
        output_dir = tmp_path / "models" / "test_vision" / "20260324T120000"
        output_dir.mkdir(parents=True, exist_ok=True)
        trainer = VisionTrainer(cfg_dict)
        full_cfg = dict(cfg_dict)
        full_cfg["_timestamp"] = "20260324T120000"
        full_cfg["_commit_hash"] = _FAKE_COMMIT
        return trainer.fit_folds(
            preprocess_output_dir=vision_fold_dir,
            output_dir=output_dir,
            cfg=full_cfg,
        )

    def test_returns_train_result(
        self, vision_fold_dir: Path, tmp_path: Path, vision_cfg: dict[str, Any]
    ) -> None:
        """fit_folds は TrainResult を返すこと。"""
        result = self._run(vision_fold_dir, tmp_path, vision_cfg)
        assert isinstance(result, TrainResult)
        assert result.trainer_type == "vision"

    def test_commit_hash_comes_from_cfg(
        self, vision_fold_dir: Path, tmp_path: Path, vision_cfg: dict[str, Any]
    ) -> None:
        """commit_hash は cfg['_commit_hash'] をそのまま使うこと。"""
        result = self._run(vision_fold_dir, tmp_path, vision_cfg)
        assert result.commit_hash == _FAKE_COMMIT

    def test_fold_results_count_matches_folds(
        self, vision_fold_dir: Path, tmp_path: Path, vision_cfg: dict[str, Any]
    ) -> None:
        """fold_results の数が実際の fold 数と一致すること。"""
        result = self._run(vision_fold_dir, tmp_path, vision_cfg)
        assert len(result.fold_results) == 1
        assert result.fold_results[0].fold_idx == 0

    def test_model_file_saved(
        self, vision_fold_dir: Path, tmp_path: Path, vision_cfg: dict[str, Any]
    ) -> None:
        """model.pt が fold ディレクトリに保存されること。"""
        result = self._run(vision_fold_dir, tmp_path, vision_cfg)
        assert result.fold_results[0].model_path.exists()
        assert result.fold_results[0].model_path.suffix == ".pt"

    def test_oof_parquet_saved(
        self, vision_fold_dir: Path, tmp_path: Path, vision_cfg: dict[str, Any]
    ) -> None:
        """oof_train.parquet が保存されること。"""
        result = self._run(vision_fold_dir, tmp_path, vision_cfg)
        assert result.fold_results[0].oof_path.exists()

    def test_error_analysis_parquet_saved(
        self, vision_fold_dir: Path, tmp_path: Path, vision_cfg: dict[str, Any]
    ) -> None:
        """error_analysis.parquet が保存されること。"""
        result = self._run(vision_fold_dir, tmp_path, vision_cfg)
        assert result.fold_results[0].error_analysis_path.exists()

    def test_seed_reproducibility(
        self, vision_fold_dir: Path, tmp_path: Path, vision_cfg: dict[str, Any]
    ) -> None:
        """同じシードで2回実行したとき CV スコアが一致すること。"""
        result1 = self._run(vision_fold_dir, tmp_path / "run1", vision_cfg)
        result2 = self._run(vision_fold_dir, tmp_path / "run2", vision_cfg)
        assert result1.cv_mean_score == pytest.approx(result2.cv_mean_score, abs=1e-4)

    def test_output_dir_differs_from_input_dir(
        self, vision_fold_dir: Path, tmp_path: Path, vision_cfg: dict[str, Any]
    ) -> None:
        """output_dir と preprocess_output_dir が異なること（再現性保証）。"""
        output_dir = tmp_path / "models" / "test_vision"
        trainer = VisionTrainer(vision_cfg)
        trainer.fit_folds(
            preprocess_output_dir=vision_fold_dir,
            output_dir=output_dir,
            cfg=vision_cfg,
        )
        assert vision_fold_dir.resolve() != output_dir.resolve()
