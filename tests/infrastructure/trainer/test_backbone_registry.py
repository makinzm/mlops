"""
Phase 2: Backbone Registry のテスト。

なぜこのテストが必要か:
  - build_backbone() が対応する全てのモデル名（resnet18/34/50, vit_b_16/32,
    mobilenet_v2/v3_small/v3_large, simple_cnn）で正しい backbone + 特徴量次元を返すことを確認する。
  - build_classifier() が正しい入力次元・出力次元の nn.Linear を返すことを確認する。
  - check_dimensions() がダミー入力で次元検証を行い、DimensionInfo を返すことを確認する。
  - 未登録の backbone 名で ValueError を送出することを確認する。

時間計算量: O(1) — 各テストは単一モデルの構築のみ
空間計算量: O(P) — P はモデルパラメータ数
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.domain.model.backbone import BackboneConfig, DimensionInfo
from src.infrastructure.trainer.backbone_registry import (
    build_backbone,
    build_classifier,
    check_dimensions,
)


class TestBuildBackbone:
    @pytest.mark.parametrize(
        "backbone_name",
        [
            "resnet18",
            "resnet34",
            "resnet50",
            "vit_b_16",
            "vit_b_32",
            "mobilenet_v2",
            "mobilenet_v3_small",
            "mobilenet_v3_large",
            "simple_cnn",
        ],
    )
    def test_build_backbone_returns_module_and_features(self, backbone_name: str) -> None:
        """各 backbone 名で nn.Module と正の特徴量次元数を返すこと。"""
        config = BackboneConfig(backbone_name=backbone_name, num_classes=2, pretrained=False)
        backbone, num_features = build_backbone(config)
        assert isinstance(backbone, nn.Module)
        assert num_features > 0

    def test_unknown_backbone_raises_value_error(self) -> None:
        """未登録の backbone 名で ValueError を送出すること。"""
        config = BackboneConfig(backbone_name="unknown_model", num_classes=2)
        with pytest.raises(ValueError, match="unknown_model"):
            build_backbone(config)


class TestBuildClassifier:
    def test_returns_linear_with_correct_dimensions(self) -> None:
        """指定した入力/出力次元の nn.Linear を返すこと。"""
        classifier = build_classifier(num_features=512, num_classes=10)
        assert isinstance(classifier, nn.Linear)
        assert classifier.in_features == 512
        assert classifier.out_features == 10


class TestCheckDimensions:
    def test_valid_dimensions_return_is_valid_true(self) -> None:
        """正しい設定で is_valid=True の DimensionInfo を返すこと。"""
        config = BackboneConfig(backbone_name="resnet18", num_classes=2, pretrained=False)
        info = check_dimensions(config)
        assert isinstance(info, DimensionInfo)
        assert info.is_valid is True

    def test_check_dimensions_with_all_backbones(self) -> None:
        """全 backbone で次元チェックが通ること。"""
        for name in [
            "resnet18",
            "resnet34",
            "resnet50",
            "vit_b_16",
            "vit_b_32",
            "mobilenet_v2",
            "mobilenet_v3_small",
            "mobilenet_v3_large",
            "simple_cnn",
        ]:
            config = BackboneConfig(backbone_name=name, num_classes=2, pretrained=False)
            info = check_dimensions(config)
            assert info.is_valid is True, f"{name}: {info.message}"
