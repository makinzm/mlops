"""
Backbone 関連のドメインデータクラス。

BackboneConfig: backbone の設定（名前、pretrained、画像サイズ、クラス数）。
DimensionInfo: backbone の出力次元チェック結果。

設計方針:
  - Domain 層のため torch/torchvision に依存しない。
  - dataclass のみで構成。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BackboneConfig:
    """Backbone の設定。

    Attributes:
        backbone_name: backbone 名（例: resnet50, vit_b_16, simple_cnn）
        num_classes: 分類クラス数
        pretrained: ImageNet 事前学習済み重みを使うか
        image_size: 入力画像サイズ（正方形）
    """

    backbone_name: str
    num_classes: int
    pretrained: bool = True
    image_size: int = 224
    custom_cnn_config: object | None = None  # CustomCNNConfig（循環 import 回避のため object）


@dataclass
class DimensionInfo:
    """Backbone の出力次元チェック結果。

    check_dimensions() がダミー入力テンソルを backbone に通し、
    出力次元と classifier の入力次元を比較した結果を保持する。

    Attributes:
        backbone_name: チェック対象の backbone 名
        output_features: backbone の実際の出力特徴量次元
        expected_features: classifier が期待する入力次元
        message: チェック結果のメッセージ
        is_valid: 次元が一致しているか
    """

    backbone_name: str
    output_features: int
    expected_features: int
    message: str
    is_valid: bool
