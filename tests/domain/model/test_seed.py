"""
SeedFixer Protocol のテスト。

なぜこのテストが必要か:
  - fix_seed() は src/infrastructure/trainer/torch_utils/seed.py に直接実装されており、
    usecase/trainer 層が PyTorch 固有の実装に直接依存していた。
  - SeedFixer Protocol を domain 層に定義し、TorchSeedFixer がそれを満たすことを保証することで、
    将来 TensorFlow/JAX 等の別フレームワーク実装に差し替え可能にする。
"""

from __future__ import annotations

from src.domain.model.seed import SeedFixer


class TestSeedFixerProtocol:
    def test_torch_seed_fixer_satisfies_protocol(self) -> None:
        """TorchSeedFixer が SeedFixer Protocol を満たすこと。"""
        from src.infrastructure.trainer.torch_utils.seed import TorchSeedFixer

        fixer: SeedFixer = TorchSeedFixer()
        assert isinstance(fixer, SeedFixer)
