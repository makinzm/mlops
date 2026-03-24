"""
Custom CNN Module — config-driven な CNN アーキテクチャ構築。

ResidualBlock: ResNet ライクな skip connection。
InvertedBottleneckBlock: MobileNet ライクな expand → depthwise → project 構造。
CustomCNNModule: CustomCNNConfig から CNN を構築する nn.Module。

output_features は __init__ 時にダミー入力で自動計算する。

時間計算量: forward は O(L * C_out * C_in * K^2 * H * W) — L: 層数
空間計算量: O(P) — P: 全パラメータ数
"""

from __future__ import annotations

import torch
from torch import nn

from src.domain.model.custom_cnn import (
    ConvBlockConfig,
    CustomCNNConfig,
)

_ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


class ResidualBlock(nn.Module):
    """ResNet ライクな Residual Block。

    入出力チャネルが異なる場合は 1x1 conv でチャネル調整する。

    時間計算量: O(C_out * C_in * K^2 * H * W)
    空間計算量: O(C_out * C_in * K^2)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut: nn.Module
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward: residual = conv → bn → relu → conv → bn; output = relu(residual + shortcut)

        時間計算量: O(C * H * W)
        空間計算量: O(C * H * W)
        """
        residual = self.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        return self.relu(residual + self.shortcut(x))


class InvertedBottleneckBlock(nn.Module):
    """MobileNet ライクな Inverted Bottleneck Block。

    expand (1x1) → depthwise (3x3) → project (1x1) の構造。
    expansion_factor で中間チャネルを拡大する。

    時間計算量: O(expand * C_in * H * W + expand * K^2 * H * W + expand * C_out * H * W)
    空間計算量: O(expand * C_in + expand * K^2 + expand * C_out)
    """

    def __init__(self, in_channels: int, out_channels: int, expansion_factor: int = 4) -> None:
        super().__init__()
        expanded = in_channels * expansion_factor
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded, 1, bias=False),
            nn.BatchNorm2d(expanded),
            nn.ReLU(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded, bias=False),
            nn.BatchNorm2d(expanded),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(expanded, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut: nn.Module
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward: expand → depthwise → project + shortcut

        時間計算量: O(C * H * W)
        空間計算量: O(C * H * W)
        """
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)
        return out + self.shortcut(x)


def _build_conv_block(in_channels: int, config: ConvBlockConfig) -> nn.Sequential:
    """ConvBlockConfig から Conv → BN → Activation → Pool のブロックを構築する。

    時間計算量: O(1) — モジュール構築のみ
    空間計算量: O(C_out * C_in * K^2)
    """
    layers: list[nn.Module] = [
        nn.Conv2d(
            in_channels,
            config.out_channels,
            config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            bias=not config.batch_norm,
        ),
    ]
    if config.batch_norm:
        layers.append(nn.BatchNorm2d(config.out_channels))

    activation_cls = _ACTIVATION_MAP.get(config.activation, nn.ReLU)
    if hasattr(activation_cls, "inplace"):
        layers.append(activation_cls(inplace=True))  # ty:ignore[invalid-argument-type]
    else:
        layers.append(activation_cls())

    if config.pool == "max":
        layers.append(nn.MaxPool2d(2))
    elif config.pool == "avg":
        layers.append(nn.AvgPool2d(2))

    return nn.Sequential(*layers)


class CustomCNNModule(nn.Module):
    """CustomCNNConfig から CNN backbone を構築する。

    output_features は __init__ 時にダミー入力 (1, 3, 32, 32) で自動計算する。

    時間計算量: forward は O(L * C * K^2 * H * W)
    空間計算量: O(P)
    """

    def __init__(self, config: CustomCNNConfig, in_channels: int = 3) -> None:
        super().__init__()
        self._config = config

        # Conv ブロックを構築
        conv_blocks: list[nn.Module] = []
        current_channels = in_channels
        channel_sizes: list[int] = [in_channels]

        for layer_cfg in config.layers:
            conv_blocks.append(_build_conv_block(current_channels, layer_cfg))
            current_channels = layer_cfg.out_channels
            channel_sizes.append(current_channels)

        self._conv_blocks = nn.ModuleList(conv_blocks)

        # Skip connections を構築
        self._skip_blocks: nn.ModuleList = nn.ModuleList()
        self._skip_indices: list[tuple[int, int]] = []

        if config.skip_connections:
            for sc in config.skip_connections:
                from_ch = channel_sizes[sc.from_layer + 1]  # from_layer の出力チャネル
                to_ch = channel_sizes[sc.to_layer + 1]  # to_layer の出力チャネル
                if sc.type == "residual":
                    self._skip_blocks.append(ResidualBlock(from_ch, to_ch))
                elif sc.type == "inverted_bottleneck":
                    self._skip_blocks.append(
                        InvertedBottleneckBlock(from_ch, to_ch, sc.expansion_factor)
                    )
                self._skip_indices.append((sc.from_layer, sc.to_layer))

        self._pool = nn.AdaptiveAvgPool2d(config.adaptive_pool_size)

        # output_features をダミー入力で計算
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 32, 32)
            out = self.forward(dummy)
            self.output_features = out.view(out.size(0), -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward: conv blocks → skip connections → adaptive pool

        時間計算量: O(L * C * K^2 * H * W)
        空間計算量: O(C * H * W)
        """
        # 各ブロックの出力を保存（skip connection 用）
        outputs: list[torch.Tensor] = []

        for i, block in enumerate(self._conv_blocks):
            x = block(x)

            # skip connection 適用: to_layer == i の場合
            for j, (from_idx, to_idx) in enumerate(self._skip_indices):
                if to_idx == i and from_idx < len(outputs):
                    skip_input = outputs[from_idx]
                    # spatial サイズを合わせる
                    if skip_input.shape[2:] != x.shape[2:]:
                        skip_input = nn.functional.adaptive_avg_pool2d(skip_input, x.shape[2:])
                    x = x + self._skip_blocks[j](skip_input)

            outputs.append(x)

        return self._pool(x)
