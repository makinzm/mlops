"""
アンサンブル戦略。

対応戦略:
- MeanStrategy          : 単純平均
- WeightedMeanStrategy  : 重み付き平均（weights は自動正規化）
- RankAverageStrategy   : rank に変換してから平均（外れ値に robust）

設計上の注意:
- 各戦略は aggregate(predictions: list[ndarray]) -> ndarray を実装する Protocol。
- WeightedMeanStrategy は weights の合計が 1.0 でなくても正規化して使用する。
- RankAverageStrategy は scipy.stats.rankdata を使い、rank を [0, 1] に正規化してから平均する。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.stats import rankdata


class EnsembleStrategy(Protocol):
    """アンサンブル戦略の Protocol。"""

    def aggregate(self, predictions: list[np.ndarray]) -> np.ndarray: ...


class MeanStrategy:
    """単純平均アンサンブル。"""

    def aggregate(self, predictions: list[np.ndarray]) -> np.ndarray:
        """予測リストの単純平均を返す。

        Args:
            predictions: 各モデルの予測値リスト（全要素の shape が同じであること）

        Returns:
            平均予測値 ndarray
        """
        return np.asarray(np.mean(np.stack(predictions, axis=0), axis=0), dtype=np.float64)


class WeightedMeanStrategy:
    """重み付き平均アンサンブル。weights は自動正規化される。"""

    def __init__(self, weights: list[float]) -> None:
        self._weights = weights

    def aggregate(self, predictions: list[np.ndarray]) -> np.ndarray:
        """予測リストの重み付き平均を返す。

        Args:
            predictions: 各モデルの予測値リスト
                         len(predictions) == len(self._weights) であること

        Returns:
            重み付き平均予測値 ndarray

        Raises:
            ValueError: weights の数と predictions の数が合わない場合
        """
        if len(predictions) != len(self._weights):
            raise ValueError(
                f"weights の数 ({len(self._weights)}) と "
                f"predictions の数 ({len(predictions)}) が一致しません"
            )
        weights = np.array(self._weights, dtype=np.float64)
        weights = weights / weights.sum()  # 正規化
        stacked = np.stack(predictions, axis=0)  # (n_models, n_samples)
        return np.asarray(np.sum(stacked * weights[:, np.newaxis], axis=0), dtype=np.float64)


class RankAverageStrategy:
    """rank average アンサンブル。外れ値に robust。

    各モデルの予測を rank に変換して [0, 1] に正規化し、その平均を返す。
    """

    def aggregate(self, predictions: list[np.ndarray]) -> np.ndarray:
        """予測リストの rank average を返す。

        Args:
            predictions: 各モデルの予測値リスト

        Returns:
            rank average 後の予測値 ndarray（[0, 1] 範囲）
        """
        ranked = []
        for pred in predictions:
            ranks = rankdata(pred)  # 1-based rank
            normalized = (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else np.zeros_like(pred)
            ranked.append(np.asarray(normalized, dtype=np.float64))
        return np.asarray(np.mean(np.stack(ranked, axis=0), axis=0), dtype=np.float64)
