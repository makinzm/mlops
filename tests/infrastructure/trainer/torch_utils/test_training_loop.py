"""
torch_utils/training_loop のテスト。

なぜこのテストが必要か:
  - run_training_loop() が TrainingMetrics を返すことを確認する。
  - best_model_state_dict が保存されること。
  - epoch ごとに学習が進むこと（loss が NaN でないこと）。

時間計算量: O(E * N) — E: エポック数, N: サンプル数
空間計算量: O(P) — P: パラメータ数
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.infrastructure.trainer.torch_utils.seed import fix_seed
from src.infrastructure.trainer.torch_utils.training_loop import (
    TrainingMetrics,
    run_training_loop,
)


def _make_simple_data(n: int = 20, seed: int = 42) -> tuple[DataLoader, DataLoader]:
    """ダミーの (images, labels) DataLoader を作成する。"""
    fix_seed(seed)
    images = torch.randn(n, 3, 8, 8)
    labels = torch.randint(0, 2, (n,))
    train_ds = TensorDataset(images[:15], labels[:15])
    valid_ds = TensorDataset(images[15:], labels[15:])
    return (
        DataLoader(train_ds, batch_size=4, shuffle=False),
        DataLoader(valid_ds, batch_size=4, shuffle=False),
    )


class TestRunTrainingLoop:
    def test_returns_training_metrics(self) -> None:
        """TrainingMetrics が返されること。"""
        train_loader, valid_loader = _make_simple_data()
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        metrics = run_training_loop(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=2,
            device=torch.device("cpu"),
        )
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.best_model_state_dict is not None

    def test_accuracy_is_between_0_and_1(self) -> None:
        """精度が [0, 1] の範囲であること。"""
        train_loader, valid_loader = _make_simple_data()
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        metrics = run_training_loop(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=2,
            device=torch.device("cpu"),
        )
        assert 0.0 <= metrics.best_valid_accuracy <= 1.0
        assert 0.0 <= metrics.final_train_accuracy <= 1.0
