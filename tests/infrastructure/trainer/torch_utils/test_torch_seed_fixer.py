"""
TorchSeedFixer のテスト。

なぜこのテストが必要か:
  - TorchSeedFixer が domain.model.seed.SeedFixer Protocol を満たすことを保証する。
  - SeedFixer Protocol 実装である TorchSeedFixer.fix() が、既存の fix_seed() 関数と
    同じ再現性を持つことを保証する（後方互換: fix_seed() 自体は削除しない）。
"""

from __future__ import annotations

import torch

from src.domain.model.seed import SeedFixer
from src.infrastructure.trainer.torch_utils.seed import TorchSeedFixer


class TestTorchSeedFixer:
    def test_satisfies_seed_fixer_protocol(self) -> None:
        """TorchSeedFixer が SeedFixer Protocol を満たすこと。"""
        fixer: SeedFixer = TorchSeedFixer()
        assert isinstance(fixer, SeedFixer)

    def test_fix_gives_torch_reproducibility(self) -> None:
        """同じ seed で fix() を呼ぶと torch の乱数が再現されること。"""
        fixer = TorchSeedFixer()
        fixer.fix(42)
        a = torch.randn(5)
        fixer.fix(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)
