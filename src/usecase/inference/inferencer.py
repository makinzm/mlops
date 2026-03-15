"""
Inferencer Protocol — 推論実行の抽象インターフェース。

UseCase 層はこの Protocol に依存し、インフラ実装（LightGBMInferencer 等）に
直接依存しない。DI は main.py で行う。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl


class Inferencer(Protocol):
    """推論実行の Protocol。

    インフラ実装例: LightGBMInferencer
    """

    def predict_folds(
        self,
        model_dir: Path,
        test_df: pl.DataFrame,
        feature_cols: list[str],
    ) -> np.ndarray:
        """全 fold のモデルで予測し、平均値を返す。

        Args:
            model_dir: fold_N/ サブディレクトリを持つモデルルートディレクトリ
            test_df: 予測対象 DataFrame
            feature_cols: 使用する特徴量カラム名リスト

        Returns:
            shape=(n_test,) の予測値 ndarray
        """
        ...
