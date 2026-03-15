"""
LightGBMInferencer — LightGBM モデルによる推論実装。

model_dir 配下の fold_N/model.lgbm を全て読み込み、
各 fold の予測値を平均して返す。

設計上の注意:
- model_dir は fold_N/ サブディレクトリを持つルートディレクトリ。
- fold_N/ の名前は "fold_" + 数字 で始まるディレクトリを自動検出。
- 複数 fold の予測値は単純平均する（アンサンブル戦略は UseCase 層で担う）。
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl


class LightGBMInferencer:
    """LightGBM モデルで fold ごとに予測し平均値を返す。"""

    MODEL_FILENAME = "model.lgbm"

    def predict_folds(
        self,
        model_dir: Path,
        test_df: pl.DataFrame,
        feature_cols: list[str],
    ) -> np.ndarray:
        """全 fold のモデルで予測し、fold 間の平均を返す。

        Args:
            model_dir: fold_N/ サブディレクトリを持つモデルルートディレクトリ
            test_df: 予測対象 DataFrame
            feature_cols: 使用する特徴量カラム名リスト

        Returns:
            shape=(n_test,) の予測値 ndarray（fold 間の平均）

        Raises:
            ValueError: fold ディレクトリが存在しない場合
        """
        fold_dirs = sorted(
            [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")],
            key=lambda d: d.name,
        )
        if not fold_dirs:
            raise ValueError(
                f"fold ディレクトリが見つかりません: {model_dir}\n"
                f"'fold_N/' という名前のサブディレクトリが必要です。"
            )

        X_test = test_df.select(feature_cols).to_pandas().values
        fold_preds: list[np.ndarray] = []

        for fold_dir in fold_dirs:
            model_path = fold_dir / self.MODEL_FILENAME
            if not model_path.exists():
                continue
            booster = lgb.Booster(model_file=str(model_path))
            pred = booster.predict(X_test)
            fold_preds.append(np.asarray(pred, dtype=np.float64))

        if not fold_preds:
            raise ValueError(
                f"有効なモデルファイルが見つかりません: {model_dir}\n"
                f"各 fold_N/ に '{self.MODEL_FILENAME}' が必要です。"
            )

        return np.asarray(np.mean(np.stack(fold_preds, axis=0), axis=0), dtype=np.float64)
