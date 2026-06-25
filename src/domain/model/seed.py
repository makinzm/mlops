"""
SeedFixer Protocol — 乱数シード固定ドメイン。

設計方針:
  - SeedFixer は Protocol。TorchSeedFixer など具体実装は infrastructure 層に置き、
    usecase/trainer は Protocol にのみ依存する。
  - フレームワーク（PyTorch/TensorFlow/JAX 等）を差し替える際、
    domain/usecase 層を変更せず infrastructure 実装だけを差し替えられるようにする。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SeedFixer(Protocol):
    """乱数シード固定の抽象 Protocol。

    TorchSeedFixer など具体実装は infrastructure/trainer/torch_utils/ に置く。
    """

    def fix(self, seed: int) -> None:
        """乱数シードを固定する。

        時間計算量: O(1)
        空間計算量: O(1)
        """
        ...
