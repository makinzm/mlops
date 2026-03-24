"""
Backbone Registry — 文字列名から torchvision モデルを構築する。

設計:
  - BACKBONE_REGISTRY: backbone 名 → コンストラクタ関数のマッピング
  - build_backbone(config) → (nn.Module, int): backbone + 出力特徴量次元
  - build_classifier(num_features, num_classes) → nn.Linear
  - check_dimensions(config) → DimensionInfo: ダミー入力で次元検証

対応 backbone:
  resnet18, resnet34, resnet50, vit_b_16, vit_b_32,
  mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, simple_cnn

時間計算量: build_backbone は O(1)（モデル構築のみ）
空間計算量: O(P) — P はモデルパラメータ数
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torchvision import models

from src.domain.model.backbone import BackboneConfig, DimensionInfo


class SimpleCNN(nn.Module):
    """テスト・軽量実験用のシンプルな CNN backbone。

    構造: Conv2d(3,16) → ReLU → MaxPool → Conv2d(16,32) → ReLU → AdaptiveAvgPool → Flatten

    時間計算量: O(C_out * C_in * K^2 * H * W) per conv layer
    空間計算量: O(C_out * C_in * K^2) パラメータ
    """

    OUTPUT_FEATURES = 32

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """フォワードパス。

        Args:
            x: (batch_size, 3, H, W) の入力テンソル

        Returns:
            (batch_size, 32, 1, 1) の特徴量テンソル
        """
        return self.features(x)


def _build_resnet(
    model_fn: Callable[..., models.ResNet], pretrained: bool
) -> tuple[nn.Module, int]:
    """ResNet 系の backbone を構築する。最終 fc 層を除去して特徴量次元を返す。

    時間計算量: O(1) — モデル構築のみ
    空間計算量: O(P) — P はモデルパラメータ数
    """
    weights = "DEFAULT" if pretrained else None
    model = model_fn(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Identity()
    return model, num_features


def _build_vit(
    model_fn: Callable[..., models.VisionTransformer], pretrained: bool
) -> tuple[nn.Module, int]:
    """ViT 系の backbone を構築する。最終 heads 層を除去して特徴量次元を返す。

    時間計算量: O(1) — モデル構築のみ
    空間計算量: O(P) — P はモデルパラメータ数
    """
    weights = "DEFAULT" if pretrained else None
    model = model_fn(weights=weights)
    num_features = model.heads.head.in_features  # ty:ignore[unresolved-attribute]
    model.heads = nn.Identity()
    return model, num_features  # ty:ignore[invalid-return-type]


def _build_mobilenet_v2(pretrained: bool) -> tuple[nn.Module, int]:
    """MobileNetV2 backbone を構築する。

    時間計算量: O(1) — モデル構築のみ
    空間計算量: O(P) — P はモデルパラメータ数
    """
    weights = "DEFAULT" if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Identity()
    return model, num_features


def _build_mobilenet_v3(
    model_fn: Callable[..., nn.Module], pretrained: bool
) -> tuple[nn.Module, int]:
    """MobileNetV3 backbone を構築する。

    時間計算量: O(1) — モデル構築のみ
    空間計算量: O(P) — P はモデルパラメータ数
    """
    weights = "DEFAULT" if pretrained else None
    model = model_fn(weights=weights)
    num_features = model.classifier[0].in_features  # ty:ignore[not-subscriptable, unresolved-attribute]
    model.classifier = nn.Identity()
    return model, num_features


def _build_simple_cnn(_pretrained: bool) -> tuple[nn.Module, int]:
    """SimpleCNN backbone を構築する。pretrained は無視される。

    時間計算量: O(1)
    空間計算量: O(P)
    """
    return SimpleCNN(), SimpleCNN.OUTPUT_FEATURES


# backbone 名 → ビルダー関数のレジストリ
_BuilderFn = Callable[[bool], tuple[nn.Module, int]]

BACKBONE_REGISTRY: dict[str, _BuilderFn] = {
    "resnet18": lambda p: _build_resnet(models.resnet18, p),
    "resnet34": lambda p: _build_resnet(models.resnet34, p),
    "resnet50": lambda p: _build_resnet(models.resnet50, p),
    "vit_b_16": lambda p: _build_vit(models.vit_b_16, p),
    "vit_b_32": lambda p: _build_vit(models.vit_b_32, p),
    "mobilenet_v2": lambda p: _build_mobilenet_v2(p),
    "mobilenet_v3_small": lambda p: _build_mobilenet_v3(models.mobilenet_v3_small, p),
    "mobilenet_v3_large": lambda p: _build_mobilenet_v3(models.mobilenet_v3_large, p),
    "simple_cnn": lambda p: _build_simple_cnn(p),
}


def build_backbone(config: BackboneConfig) -> tuple[nn.Module, int]:
    """BackboneConfig から backbone を構築し、(module, 出力特徴量次元) を返す。

    時間計算量: O(1) — モデル構築のみ
    空間計算量: O(P) — P はモデルパラメータ数

    Raises:
        ValueError: 未登録の backbone 名の場合
    """
    name = config.backbone_name
    if name not in BACKBONE_REGISTRY:
        registered = sorted(BACKBONE_REGISTRY.keys())
        raise ValueError(f"未登録の backbone: '{name}'. 登録済み: {registered}")
    return BACKBONE_REGISTRY[name](config.pretrained)


def build_classifier(num_features: int, num_classes: int) -> nn.Linear:
    """分類ヘッドの nn.Linear を構築する。

    時間計算量: O(num_features * num_classes)
    空間計算量: O(num_features * num_classes)
    """
    return nn.Linear(num_features, num_classes)


def check_dimensions(config: BackboneConfig) -> DimensionInfo:
    """ダミー入力で backbone の出力次元を検証する。

    backbone にダミーテンソル (1, 3, image_size, image_size) を通し、
    出力の特徴量次元が build_backbone が返す次元と一致するか確認する。

    時間計算量: O(P) — 1 回のフォワードパス
    空間計算量: O(P + image_size^2)
    """
    backbone, expected_features = build_backbone(config)
    backbone.eval()

    dummy_input = torch.zeros(1, 3, config.image_size, config.image_size)
    with torch.no_grad():
        output = backbone(dummy_input)

    # Flatten して特徴量次元を取得
    actual_features = output.view(output.size(0), -1).size(1)

    if actual_features == expected_features:
        return DimensionInfo(
            backbone_name=config.backbone_name,
            output_features=actual_features,
            expected_features=expected_features,
            message="OK",
            is_valid=True,
        )
    else:
        return DimensionInfo(
            backbone_name=config.backbone_name,
            output_features=actual_features,
            expected_features=expected_features,
            message=(
                f"Dimension mismatch: backbone outputs {actual_features} features "
                f"but expected {expected_features}"
            ),
            is_valid=False,
        )
