"""
Custom CNN のドメインデータクラス。

ConvBlockConfig: 1 つの Conv ブロックの設定。
SkipConnectionConfig: Skip connection（Residual / Inverted Bottleneck）の設定。
CustomCNNConfig: CNN 全体の構造設定。

設計方針:
  - Domain 層のため torch に依存しない。
  - dataclass のみで構成。infrastructure 層の CustomCNNModule が実際のモデルを構築する。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConvBlockConfig:
    """1 つの Conv ブロックの設定。

    Attributes:
        out_channels: 出力チャネル数
        kernel_size: カーネルサイズ
        stride: ストライド
        padding: パディング
        activation: 活性化関数（"relu", "silu", "gelu"）
        batch_norm: BatchNorm を使うか
        pool: プーリング種別（"max", "avg", None）
    """

    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    activation: str = "relu"
    batch_norm: bool = True
    pool: str | None = "max"


@dataclass
class SkipConnectionConfig:
    """Skip connection の設定。

    Attributes:
        type: "residual" or "inverted_bottleneck"
        from_layer: 接続元のレイヤーインデックス
        to_layer: 接続先のレイヤーインデックス
        expansion_factor: inverted bottleneck の拡張係数
    """

    type: str  # "residual" | "inverted_bottleneck"
    from_layer: int = 0
    to_layer: int = 1
    expansion_factor: int = 1


@dataclass
class CustomCNNConfig:
    """Custom CNN 全体の構造設定。

    Attributes:
        layers: Conv ブロック設定のリスト
        skip_connections: Skip connection 設定のリスト（None なら skip なし）
        adaptive_pool_size: 最終 AdaptiveAvgPool2d の出力サイズ
    """

    layers: list[ConvBlockConfig]
    skip_connections: list[SkipConnectionConfig] | None = None
    adaptive_pool_size: int = 1
