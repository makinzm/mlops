"""
torch_utils/seed のテスト。

なぜこのテストが必要か:
  - fix_seed() が torch, numpy, cudnn の乱数状態を固定することを確認する。
  - 同じ seed で2回呼び出した後、同じ乱数列が得られることを保証する。
  - 音声・言語モデルでも再利用される基盤なので独立テストが必要。

時間計算量: O(1)
空間計算量: O(1)
"""

from __future__ import annotations

import numpy as np
import torch

from src.infrastructure.trainer.torch_utils.seed import fix_seed


class TestFixSeed:
    def test_torch_reproducibility(self) -> None:
        """同じ seed で torch の乱数が再現されること。"""
        fix_seed(42)
        a = torch.randn(5)
        fix_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)

    def test_numpy_reproducibility(self) -> None:
        """同じ seed で numpy の乱数が再現されること。"""
        fix_seed(42)
        a = np.random.rand(5)
        fix_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        """異なる seed で異なる乱数列が生成されること。"""
        fix_seed(42)
        a = torch.randn(5)
        fix_seed(99)
        b = torch.randn(5)
        assert not torch.allclose(a, b)
