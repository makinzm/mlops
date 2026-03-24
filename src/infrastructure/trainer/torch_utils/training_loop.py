"""
汎用 PyTorch 学習ループ。

Vision・音声・言語モデルなど全ての PyTorch ベースの学習で共通利用する。
モデル・データローダー・損失関数・オプティマイザを受け取り、
epoch 反復 → train step → valid step → ベストモデル保存 を行う。

時間計算量: O(E * N * C) — E: エポック, N: サンプル, C: モデル計算量
空間計算量: O(P + B * C * H * W) — P: パラメータ, B: バッチサイズ
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """学習ループの結果メトリクス。

    Attributes:
        best_valid_accuracy: validation の最高精度
        final_train_accuracy: 最終 epoch の train 精度
        best_model_state_dict: ベストモデルの state_dict
    """

    best_valid_accuracy: float
    final_train_accuracy: float
    best_model_state_dict: OrderedDict[str, torch.Tensor]


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,  # ty:ignore[invalid-argument-type]
    valid_loader: DataLoader,  # ty:ignore[invalid-argument-type]
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
) -> TrainingMetrics:
    """PyTorch の学習ループを実行する。

    Args:
        model: 学習対象モデル
        train_loader: 訓練データの DataLoader
        valid_loader: 検証データの DataLoader
        criterion: 損失関数
        optimizer: オプティマイザ
        num_epochs: エポック数
        device: 計算デバイス

    Returns:
        TrainingMetrics: 学習結果のメトリクス

    時間計算量: O(E * N * C)
    空間計算量: O(P)
    """
    best_valid_acc = 0.0
    best_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    final_train_acc = 0.0

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / max(train_total, 1)
        final_train_acc = train_acc

        # Validation
        model.eval()
        valid_correct = 0
        valid_total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                valid_correct += predicted.eq(labels).sum().item()
                valid_total += labels.size(0)

        valid_acc = valid_correct / max(valid_total, 1)

        if valid_acc >= best_valid_acc:
            best_valid_acc = valid_acc
            best_state = OrderedDict({k: v.clone() for k, v in model.state_dict().items()})

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs}: train_acc={train_acc:.4f}, valid_acc={valid_acc:.4f}"
        )

    return TrainingMetrics(
        best_valid_accuracy=best_valid_acc,
        final_train_accuracy=final_train_acc,
        best_model_state_dict=best_state,
    )
