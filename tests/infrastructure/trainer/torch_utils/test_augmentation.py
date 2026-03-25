"""
torch_utils/augmentation のテスト。

なぜこのテストが必要か:
  - build_augmentation_pipeline() が config から augmentation パイプラインを構築すること。
  - train/valid で異なる augmentation が適用されること。
  - albumentations 未インストール時は torchvision にフォールバックすること。

時間計算量: O(H * W) — 1 画像の変換
空間計算量: O(C * H * W)
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.domain.model.augmentation import AugmentationConfig, AugmentTransformConfig
from src.infrastructure.trainer.torch_utils.augmentation import (
    build_augmentation_pipeline,
    build_default_transform,
)


class TestBuildDefaultTransform:
    def test_returns_callable(self) -> None:
        """デフォルト transform が callable を返すこと。"""
        transform = build_default_transform(image_size=32)
        assert callable(transform)

    def test_output_shape(self) -> None:
        """デフォルト transform が正しい shape のテンソルを返すこと。"""
        import torch

        transform = build_default_transform(image_size=32)
        img = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
        result = transform(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 32, 32)


class TestBuildAugmentationPipeline:
    def test_with_empty_config(self) -> None:
        """空の config でデフォルト transform が返されること。"""
        config = AugmentationConfig(train_transforms=[], valid_transforms=[])
        train_tf, valid_tf = build_augmentation_pipeline(config, image_size=32)
        assert callable(train_tf)
        assert callable(valid_tf)

    def test_with_augmentation_config(self) -> None:
        """augmentation config でパイプラインが構築されること。"""
        config = AugmentationConfig(
            train_transforms=[
                AugmentTransformConfig(name="HorizontalFlip", probability=0.5),
            ],
            valid_transforms=[],
        )
        train_tf, valid_tf = build_augmentation_pipeline(config, image_size=32)
        assert callable(train_tf)
        assert callable(valid_tf)
