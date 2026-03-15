"""
EnsembleStrategy の単体テスト。

なぜこのテストが必要か:
  - MeanStrategy / WeightedMeanStrategy / RankAverageStrategy は Inference の中核。
    各戦略が数値的に正しいことをテストで保証しないと、
    アンサンブル結果の誤りに気づけない。
  - rank_average は確率ではなくランク順位を平均するため、
    外れ値に対して robust な特性を持つ。
    この変換が正しく行われていることを数値で確認する必要がある。
  - weighted_mean は weights の合計が 1.0 でなくても正規化されること（使いやすさ）を
    テストで明示する。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.usecase.inference.ensemble_strategies import (
    MeanStrategy,
    RankAverageStrategy,
    WeightedMeanStrategy,
)


class TestMeanStrategy:
    def test_mean_strategy(self) -> None:
        """2 つの予測を単純平均すること。"""
        strategy = MeanStrategy()
        preds = [
            np.array([0.2, 0.4, 0.6]),
            np.array([0.4, 0.6, 0.8]),
        ]
        result = strategy.aggregate(preds)
        expected = np.array([0.3, 0.5, 0.7])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_mean_strategy_single_pred(self) -> None:
        """1 つの予測のみの場合はそのまま返すこと。"""
        strategy = MeanStrategy()
        preds = [np.array([0.3, 0.7])]
        result = strategy.aggregate(preds)
        np.testing.assert_allclose(result, np.array([0.3, 0.7]), atol=1e-6)


class TestWeightedMeanStrategy:
    def test_weighted_mean_strategy(self) -> None:
        """weights=[0.6, 0.4] で加重平均になること。"""
        strategy = WeightedMeanStrategy(weights=[0.6, 0.4])
        preds = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
        ]
        result = strategy.aggregate(preds)
        expected = np.array([0.6, 0.4])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_weighted_mean_normalizes_weights(self) -> None:
        """weights の合計が 1.0 でなくても正規化されること。"""
        strategy = WeightedMeanStrategy(weights=[3.0, 1.0])  # 合計 4.0 → 0.75, 0.25 に正規化
        preds = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
        ]
        result = strategy.aggregate(preds)
        expected = np.array([0.75, 0.25])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_weighted_mean_raises_on_weight_mismatch(self) -> None:
        """weights の数と予測数が合わない場合に ValueError を送出すること。"""
        strategy = WeightedMeanStrategy(weights=[0.5, 0.3, 0.2])
        preds = [np.array([0.5]), np.array([0.5])]
        with pytest.raises(ValueError, match="weights"):
            strategy.aggregate(preds)


class TestRankAverageStrategy:
    def test_rank_average_strategy(self) -> None:
        """rank に変換してから平均すること。

        予測値 [0.9, 0.1, 0.5] の rank は [3, 1, 2]。
        2 モデルが同じ予測なら rank_average も同じ値になる。
        """
        strategy = RankAverageStrategy()
        # モデル1, モデル2 が同じ予測
        pred = np.array([0.9, 0.1, 0.5])
        preds = [pred.copy(), pred.copy()]
        result = strategy.aggregate(preds)
        # rank は [3, 1, 2] → 正規化 → [1.0, 0.0, 0.5]
        # 2 モデルで平均しても同じ
        assert result[0] > result[2] > result[1]  # 相対順序が保たれること

    def test_rank_average_is_robust_to_outlier(self) -> None:
        """外れ値に対して rank_average が robust であること。

        モデル1: [0.99, 0.5, 0.1] — index 0 が高予測
        モデル2: [0.5, 0.99, 0.1] — index 1 が高予測
        rank_average は各モデルで rank に変換してから平均するため、
        絶対値の差異が吸収される。
        """
        strategy = RankAverageStrategy()
        preds = [
            np.array([0.99, 0.5, 0.1]),
            np.array([0.5, 0.99, 0.1]),
        ]
        result = strategy.aggregate(preds)
        # index 0 と index 1 が index 2 より高いこと
        assert result[0] > result[2]
        assert result[1] > result[2]
