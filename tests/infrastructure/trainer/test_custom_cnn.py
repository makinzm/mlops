"""
Custom CNN のテスト。

なぜこのテストが必要か:
  - CustomCNNModule が config からモデルを構築し、正しい出力次元を返すこと。
  - ResidualBlock が skip connection を正しく適用すること。
  - InvertedBottleneckBlock が expand → depthwise → project の構造を持つこと。
  - register_backbone() でカスタム backbone を登録し、build_backbone() で構築できること。

時間計算量: O(C * H * W) per forward pass
空間計算量: O(P) — P: パラメータ数
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.domain.model.backbone import BackboneConfig
from src.domain.model.custom_cnn import (
    ConvBlockConfig,
    CustomCNNConfig,
    SkipConnectionConfig,
)
from src.infrastructure.trainer.backbone_registry import (
    build_backbone,
    register_backbone,
)
from src.infrastructure.trainer.custom_cnn import (
    CustomCNNModule,
    InvertedBottleneckBlock,
    ResidualBlock,
)


class TestResidualBlock:
    def test_output_shape_matches_input(self) -> None:
        """入出力チャネルが同じ場合、shape が一致すること。"""
        block = ResidualBlock(in_channels=32, out_channels=32)
        x = torch.randn(1, 32, 16, 16)
        y = block(x)
        assert y.shape == x.shape

    def test_output_shape_with_channel_change(self) -> None:
        """入出力チャネルが異なる場合、1x1 conv でチャネル調整されること。"""
        block = ResidualBlock(in_channels=16, out_channels=32)
        x = torch.randn(1, 16, 16, 16)
        y = block(x)
        assert y.shape == (1, 32, 16, 16)


class TestInvertedBottleneckBlock:
    def test_output_shape(self) -> None:
        """expand → depthwise → project で正しい出力 shape を返すこと。"""
        block = InvertedBottleneckBlock(in_channels=16, out_channels=32, expansion_factor=4)
        x = torch.randn(1, 16, 16, 16)
        y = block(x)
        assert y.shape == (1, 32, 16, 16)

    def test_expansion_factor(self) -> None:
        """expansion_factor で中間チャネルが拡大されること。"""
        block = InvertedBottleneckBlock(in_channels=16, out_channels=16, expansion_factor=6)
        # 内部の expand conv は 16 → 96 チャネル
        x = torch.randn(1, 16, 8, 8)
        y = block(x)
        assert y.shape == (1, 16, 8, 8)


class TestCustomCNNModule:
    def test_basic_config(self) -> None:
        """基本的な config でモデルが構築されること。"""
        config = CustomCNNConfig(
            layers=[
                ConvBlockConfig(out_channels=16, kernel_size=3),
                ConvBlockConfig(out_channels=32, kernel_size=3),
            ],
        )
        model = CustomCNNModule(config)
        x = torch.randn(1, 3, 32, 32)
        features = model(x)
        assert features.shape[0] == 1
        assert model.output_features > 0

    def test_with_residual_skip(self) -> None:
        """Residual skip connection 付き config でモデルが構築されること。"""
        config = CustomCNNConfig(
            layers=[
                ConvBlockConfig(out_channels=32, kernel_size=3),
                ConvBlockConfig(out_channels=32, kernel_size=3, pool=None),
                ConvBlockConfig(out_channels=64, kernel_size=3),
            ],
            skip_connections=[
                SkipConnectionConfig(type="residual", from_layer=0, to_layer=1),
            ],
        )
        model = CustomCNNModule(config)
        x = torch.randn(1, 3, 32, 32)
        features = model(x)
        assert features.shape[0] == 1
        assert model.output_features > 0

    def test_with_inverted_bottleneck(self) -> None:
        """Inverted bottleneck 付き config でモデルが構築されること。"""
        config = CustomCNNConfig(
            layers=[
                ConvBlockConfig(out_channels=16, kernel_size=3),
                ConvBlockConfig(out_channels=32, kernel_size=3),
            ],
            skip_connections=[
                SkipConnectionConfig(
                    type="inverted_bottleneck", from_layer=0, to_layer=1, expansion_factor=4
                ),
            ],
        )
        model = CustomCNNModule(config)
        x = torch.randn(1, 3, 32, 32)
        features = model(x)
        assert features.shape[0] == 1


class TestRegisterBackbone:
    def test_register_and_build(self) -> None:
        """register_backbone() で登録した backbone が build_backbone() で構築できること。"""
        def my_builder(pretrained: bool) -> tuple[nn.Module, int]:
            return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.AdaptiveAvgPool2d(1)), 8

        register_backbone("test_custom_backbone", my_builder)
        config = BackboneConfig(backbone_name="test_custom_backbone", num_classes=2)
        backbone, features = build_backbone(config)
        assert isinstance(backbone, nn.Module)
        assert features == 8

    def test_custom_cnn_backbone_from_config(self) -> None:
        """custom_cnn backbone が BackboneConfig + CustomCNNConfig で構築できること。"""
        custom_config = CustomCNNConfig(
            layers=[
                ConvBlockConfig(out_channels=16, kernel_size=3),
                ConvBlockConfig(out_channels=32, kernel_size=3),
            ],
        )
        config = BackboneConfig(
            backbone_name="custom_cnn",
            num_classes=2,
            pretrained=False,
            image_size=32,
            custom_cnn_config=custom_config,
        )
        backbone, features = build_backbone(config)
        assert isinstance(backbone, nn.Module)
        assert features > 0
