"""
LightGBMInferencer の統合テスト。

なぜこのテストが必要か:
  - LightGBMInferencer は fold_N/model.lgbm を順番に読み込んで
    test_df に対して予測を行う。
  - 実際の lgbm モデルファイルを使って predict が動くことを確認しないと、
    モデルファイルの形式が変わったときに気づけない。
  - 複数 fold の平均が正しく計算されることをテストで保証する。
  - output が shape=(n_test,) の ndarray であることを確認する。
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

from src.infrastructure.inference.lgbm_inferencer import LightGBMInferencer

# ──────────────────────────────────────────────────────────────
# フィクスチャ
# ──────────────────────────────────────────────────────────────


def _make_lgbm_model(path: Path, seed: int = 42) -> None:
    """小さな LightGBM モデルを訓練して path に保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = 50
    X = rng.uniform(0, 1, (n, 3))
    y = rng.integers(0, 2, n).astype(float)

    train_data = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 4,
        "n_estimators": 10,
        "verbose": -1,
        "seed": seed,
    }
    booster = lgb.train(params, train_data, num_boost_round=10)
    booster.save_model(str(path))


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """fold_0, fold_1 に lgbm モデルを持つディレクトリ。"""
    for fold_idx in range(2):
        model_path = tmp_path / f"fold_{fold_idx}" / "model.lgbm"
        _make_lgbm_model(model_path, seed=fold_idx)
    return tmp_path


@pytest.fixture
def test_df() -> pl.DataFrame:
    """予測対象の test DataFrame（5 サンプル）。"""
    rng = np.random.default_rng(99)
    return pl.DataFrame(
        {
            "PassengerId": list(range(892, 897)),
            "feat_0": rng.uniform(0, 1, 5).tolist(),
            "feat_1": rng.uniform(0, 1, 5).tolist(),
            "feat_2": rng.uniform(0, 1, 5).tolist(),
        }
    )


# ──────────────────────────────────────────────────────────────
# テスト
# ──────────────────────────────────────────────────────────────


class TestLightGBMInferencer:
    def test_predict_folds_returns_ndarray(self, model_dir: Path, test_df: pl.DataFrame) -> None:
        """predict_folds が shape=(n_test,) の ndarray を返すこと。"""
        inferencer = LightGBMInferencer()
        feature_cols = ["feat_0", "feat_1", "feat_2"]
        result = inferencer.predict_folds(model_dir, test_df, feature_cols)

        assert isinstance(result, np.ndarray), f"ndarray を期待したが {type(result)} が返った"
        assert result.shape == (5,), f"shape=(5,) を期待したが {result.shape} が返った"

    def test_predict_folds_values_in_probability_range(
        self, model_dir: Path, test_df: pl.DataFrame
    ) -> None:
        """binary 分類の予測値が [0, 1] の範囲にあること。"""
        inferencer = LightGBMInferencer()
        feature_cols = ["feat_0", "feat_1", "feat_2"]
        result = inferencer.predict_folds(model_dir, test_df, feature_cols)

        assert np.all(result >= 0.0) and np.all(result <= 1.0), (
            f"予測値が [0, 1] 範囲外: min={result.min():.4f}, max={result.max():.4f}"
        )

    def test_predict_folds_averages_over_folds(self, tmp_path: Path) -> None:
        """複数 fold の予測が平均されること。

        fold_0 と fold_1 が同じモデルであれば、平均も同じ値になる。
        fold_0 のみの予測と fold_0+fold_1 の平均が一致することで
        average ロジックを検証する。
        """
        # 同一モデルを 2 fold に配置
        for fold_idx in range(2):
            model_path = tmp_path / f"fold_{fold_idx}" / "model.lgbm"
            _make_lgbm_model(model_path, seed=0)  # 同じ seed で同じモデル

        rng = np.random.default_rng(99)
        test_df_local = pl.DataFrame(
            {
                "f0": rng.uniform(0, 1, 5).tolist(),
                "f1": rng.uniform(0, 1, 5).tolist(),
                "f2": rng.uniform(0, 1, 5).tolist(),
            }
        )
        feature_cols = ["f0", "f1", "f2"]

        # 2 fold の average
        inferencer = LightGBMInferencer()
        result_2fold = inferencer.predict_folds(tmp_path, test_df_local, feature_cols)

        # fold_0 のみのディレクトリ
        single_dir = tmp_path / "single_fold_dir"
        single_dir.mkdir()
        _make_lgbm_model(single_dir / "fold_0" / "model.lgbm", seed=0)
        result_1fold = inferencer.predict_folds(single_dir, test_df_local, feature_cols)

        # 同じモデルなので平均しても同じ値になるはず
        np.testing.assert_allclose(result_2fold, result_1fold, atol=1e-6)

    def test_predict_folds_raises_when_no_model_dir(self, tmp_path: Path) -> None:
        """fold ディレクトリが存在しない場合に ValueError を送出すること。"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        test_df_local = pl.DataFrame({"f0": [0.5]})

        inferencer = LightGBMInferencer()
        with pytest.raises(ValueError, match="fold"):
            inferencer.predict_folds(empty_dir, test_df_local, ["f0"])
