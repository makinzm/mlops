"""
Data Augmentation のドメインデータクラス。

AugmentTransformConfig: 1 つの augmentation transform の設定。
AugmentationConfig: train/valid 別の augmentation パイプライン設定。

設計方針:
  - Domain 層のため albumentations/torchvision に依存しない。
  - dataclass のみで構成。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AugmentTransformConfig:
    """1 つの augmentation transform の設定。

    Attributes:
        name: transform 名（例: "HorizontalFlip", "RandomBrightnessContrast"）
        probability: 適用確率
        params: transform 固有のパラメータ
    """

    name: str
    probability: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentationConfig:
    """train/valid 別の augmentation パイプライン設定。

    Attributes:
        train_transforms: 訓練時に適用する transform リスト
        valid_transforms: 検証時に適用する transform リスト（通常は空）
    """

    train_transforms: list[AugmentTransformConfig]
    valid_transforms: list[AugmentTransformConfig]
