"""
Vision モデルの構築・チェックポイント管理。

build_vision_model / save_checkpoint / load_checkpoint を提供し、
VisionTrainer / VisionInferencer / GradCAMAnalyzer で統一的に使う。

チェックポイント形式:
  {
    "model_state_dict": OrderedDict,
    "backbone_config": {
      "backbone_name": str,
      "num_classes": int,
      "pretrained": bool,
      "image_size": int,
    },
  }

時間計算量: build は O(P), save/load は O(P) — P: パラメータ数
空間計算量: O(P)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.domain.model.backbone import BackboneConfig
from src.infrastructure.trainer.backbone_registry import build_backbone, build_classifier


def build_vision_model(config: BackboneConfig) -> nn.Module:
    """BackboneConfig から backbone + flatten + classifier の nn.Sequential を構築する。

    Args:
        config: backbone 設定

    Returns:
        nn.Sequential モデル

    時間計算量: O(P)
    空間計算量: O(P)
    """
    backbone, num_features = build_backbone(config)
    classifier = build_classifier(num_features, config.num_classes)
    return nn.Sequential(backbone, nn.Flatten(), classifier)


def save_checkpoint(model: nn.Module, config: BackboneConfig, path: Path) -> None:
    """モデルとメタ情報をチェックポイントとして保存する。

    Args:
        model: 保存するモデル
        config: backbone 設定（復元に必要）
        path: 保存先パス

    時間計算量: O(P)
    空間計算量: O(P)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    config_dict: dict[str, Any] = {
        "backbone_name": config.backbone_name,
        "num_classes": config.num_classes,
        "pretrained": config.pretrained,
        "image_size": config.image_size,
    }
    if config.custom_cnn_config is not None:
        config_dict["custom_cnn_config"] = _serialize_custom_cnn_config(config.custom_cnn_config)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "backbone_config": config_dict,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device | None = None) -> nn.Module:
    """チェックポイントからモデルを復元する。

    Args:
        path: チェックポイントのパス
        device: 計算デバイス（None の場合は CPU）

    Returns:
        復元された nn.Module

    時間計算量: O(P)
    空間計算量: O(P)
    """
    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    backbone_cfg = checkpoint["backbone_config"]

    custom_cnn_config = None
    if "custom_cnn_config" in backbone_cfg:
        custom_cnn_config = _deserialize_custom_cnn_config(backbone_cfg["custom_cnn_config"])

    config = BackboneConfig(
        backbone_name=backbone_cfg["backbone_name"],
        num_classes=backbone_cfg["num_classes"],
        pretrained=False,
        image_size=backbone_cfg.get("image_size", 32),
        custom_cnn_config=custom_cnn_config,
    )

    model = build_vision_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _serialize_custom_cnn_config(config: object) -> dict:
    """CustomCNNConfig を dict に変換する。"""
    from dataclasses import asdict

    return asdict(config)  # ty:ignore[invalid-argument-type]


def _deserialize_custom_cnn_config(data: dict) -> object:
    """dict から CustomCNNConfig を復元する。"""
    from src.domain.model.custom_cnn import (
        ConvBlockConfig,
        CustomCNNConfig,
        SkipConnectionConfig,
    )

    layers = [ConvBlockConfig(**layer) for layer in data["layers"]]
    skip_connections = None
    if data.get("skip_connections"):
        skip_connections = [SkipConnectionConfig(**sc) for sc in data["skip_connections"]]
    return CustomCNNConfig(
        layers=layers,
        skip_connections=skip_connections,
        adaptive_pool_size=data.get("adaptive_pool_size", 1),
    )
