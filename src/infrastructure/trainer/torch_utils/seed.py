"""
乱数シード固定ユーティリティ。

PyTorch / NumPy / cuDNN の乱数状態を固定して再現性を保証する。
Vision・音声・言語モデルなど全ての PyTorch ベースの学習で共通利用する。

時間計算量: O(1)
空間計算量: O(1)
"""

from __future__ import annotations

import numpy as np
import torch


def fix_seed(seed: int) -> None:
    """PyTorch / NumPy / cuDNN の乱数状態を固定する。

    Args:
        seed: 乱数シード値

    時間計算量: O(1)
    空間計算量: O(1)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TorchSeedFixer:
    """domain.model.seed.SeedFixer Protocol の PyTorch 実装。

    fix_seed() をそのまま呼ぶ薄いラッパー。usecase/trainer 層が
    PyTorch 固有の実装に直接依存しないよう Protocol 越しに使う。
    """

    def fix(self, seed: int) -> None:
        fix_seed(seed)
