"""
Phase 4: VisionInferencer の統合テスト。

なぜこのテストが必要か:
  - VisionInferencer は Inferencer Protocol を満たし、predict_folds() が
    fold ごとの model.pt を読み込んで推論 → fold 平均を返すことを確認する。
  - 合成画像（8x8 PNG）を使って実際に推論が動くことを確認する。
  - 出力が shape=(n_test,) の ndarray であること。
  - 予測値が [0, 1] の範囲にあること（softmax 出力）。
  - fold ディレクトリが存在しない場合に ValueError を送出すること。

時間計算量: O(F * N) — F: fold 数, N: テストサンプル数
空間計算量: O(P + N) — P: モデルパラメータ数, N: テストサンプル数
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from PIL import Image
from torch import nn

from src.domain.model.backbone import BackboneConfig
from src.infrastructure.inference.vision_inferencer import VisionInferencer
from src.infrastructure.trainer.backbone_registry import build_backbone, build_classifier


def _create_synthetic_images(image_dir: Path, num_images: int, seed: int = 42) -> list[str]:
    """テスト用合成画像を生成する。"""
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths: list[str] = []
    for i in range(num_images):
        data = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        img = Image.fromarray(data)
        path = image_dir / f"test_{i:04d}.png"
        img.save(path)
        paths.append(str(path))
    return paths


def _make_vision_model(path: Path, num_classes: int = 2, seed: int = 42) -> None:
    """テスト用の小さい Vision モデルを保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    config = BackboneConfig(backbone_name="simple_cnn", num_classes=num_classes, pretrained=False)
    backbone, num_features = build_backbone(config)
    classifier = build_classifier(num_features, num_classes)
    model = nn.Sequential(backbone, nn.Flatten(), classifier)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "backbone_config": {
                "backbone_name": "simple_cnn",
                "num_classes": num_classes,
                "pretrained": False,
                "image_size": 32,
            },
        },
        path,
    )


@pytest.fixture
def vision_model_dir(tmp_path: Path) -> Path:
    """fold_0, fold_1 に vision モデルを持つディレクトリ。"""
    for fold_idx in range(2):
        model_path = tmp_path / f"fold_{fold_idx}" / "model.pt"
        _make_vision_model(model_path, seed=fold_idx)
    return tmp_path


@pytest.fixture
def test_image_df(tmp_path: Path) -> pl.DataFrame:
    """予測対象の test DataFrame（5 サンプル、合成画像）。"""
    paths = _create_synthetic_images(tmp_path / "test_images", 5, seed=99)
    return pl.DataFrame({"image_path": paths})


class TestVisionInferencer:
    def test_predict_folds_returns_ndarray(
        self, vision_model_dir: Path, test_image_df: pl.DataFrame
    ) -> None:
        """predict_folds が shape=(n_test,) の ndarray を返すこと。"""
        inferencer = VisionInferencer()
        result = inferencer.predict_folds(vision_model_dir, test_image_df, ["image_path"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)

    def test_predict_folds_values_in_probability_range(
        self, vision_model_dir: Path, test_image_df: pl.DataFrame
    ) -> None:
        """予測値が [0, 1] の範囲にあること。"""
        inferencer = VisionInferencer()
        result = inferencer.predict_folds(vision_model_dir, test_image_df, ["image_path"])
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_predict_folds_raises_when_no_model_dir(self, tmp_path: Path) -> None:
        """fold ディレクトリが存在しない場合に ValueError を送出すること。"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        test_df = pl.DataFrame({"image_path": ["/dummy.png"]})

        inferencer = VisionInferencer()
        with pytest.raises(ValueError, match="fold"):
            inferencer.predict_folds(empty_dir, test_df, ["image_path"])
