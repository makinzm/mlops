"""
SeedFixer Protocol のテスト。

なぜこのテストが必要か:
  - SeedFixer Protocol を domain 層に定義し、fix(seed: int) -> None を満たす
    任意のオブジェクトが Protocol を満たすことを保証する。
  - 具体実装（TorchSeedFixer 等）への依存は infrastructure 層のテストで検証する
    （Clean Architecture: tests/domain は src/infrastructure に依存してはならない）。
"""

from __future__ import annotations

from src.domain.model.seed import SeedFixer


class _FakeSeedFixer:
    def fix(self, seed: int) -> None:
        pass


class TestSeedFixerProtocol:
    def test_object_with_fix_method_satisfies_protocol(self) -> None:
        """fix(seed: int) -> None を実装したオブジェクトが Protocol を満たすこと。"""
        fixer: SeedFixer = _FakeSeedFixer()
        assert isinstance(fixer, SeedFixer)
