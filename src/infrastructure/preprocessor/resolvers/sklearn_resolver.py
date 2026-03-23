"""
sklearn ベースの変換 Resolver。

対応メソッド:
- fill_na      : 欠損補完（median/mean/constant）
                 Train で fit した統計量を Test に transform することでデータリークを防ぐ。
- target_encode: OOF（Out-of-Fold）CV ベースの Target Encoding。
                 val fold のデータを encoder の fit に使わないことでデータリークを防ぐ。
                 smoothing で小カテゴリの過学習を防ぐ。

設計上の注意:
- fill_na は (transformed_df, fitted_imputer) のタプルを返す。
  DAGRunner は fitted_imputer を保持し、Test データには transform() を呼ぶ。
- Polars の null を pandas 経由で SimpleImputer に渡し、結果を Polars に戻す。
- target_encode は (oof_encoded_df, full_encoder) のタプルを返す。
  full_encoder は全 train で fit した encoder。Test には transform_target_encode() を使う。
  full_encoder[col]["__global_mean__"] が未知カテゴリのフォールバック値。
"""

from typing import Any

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

# encoder の型エイリアス: {col: {"category": encoded_value, "__global_mean__": float}}
TargetEncoder = dict[str, dict[str, float]]


class SklearnResolver:
    """sklearn を使ったタブラーデータ変換 Resolver。"""

    def supported_methods(self) -> set[str]:
        return {"fill_na", "target_encode"}

    def execute(self, df: pl.DataFrame, method: str, **kwargs: object) -> pl.DataFrame:
        """メソッド名に応じて変換を実行する。

        注意: fill_na は (df, imputer) タプルを返すため、
        DAGRunner は execute() ではなく直接 fill_na() を呼ぶことを推奨する。
        """
        if method == "fill_na":
            strategy = str(kwargs.get("strategy", "median"))
            columns = kwargs.get("columns")
            if not isinstance(columns, list):
                raise ValueError("fill_na requires 'columns' as list[str]")
            fill_value = kwargs.get("fill_value", 0.0)
            result, _ = self.fill_na(df, strategy=strategy, columns=columns, fill_value=fill_value)  # ty:ignore[invalid-argument-type]
            return result
        raise ValueError(f"Unknown method: {method!r}")

    def fill_na(
        self,
        df: pl.DataFrame,
        strategy: str,
        columns: list[str],
        fill_value: Any = 0.0,
    ) -> tuple[pl.DataFrame, SimpleImputer]:
        """欠損値補完を行い、(変換後DataFrame, fittedImputer) を返す。

        Args:
            df: 変換対象 DataFrame（Train データで呼ぶことを想定）
            strategy: "median" / "mean" / "constant"
            columns: 補完対象カラム名リスト
            fill_value: strategy="constant" のときに使う埋め値

        Returns:
            (transformed_df, fitted_imputer)
            fitted_imputer を保持して transform() に渡すことで
            Test データのリークを防ぐ。
        """
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        subset = df.select(columns).to_pandas()
        imputer.fit(subset)
        transformed = np.asarray(imputer.transform(subset))

        result = df.clone()
        for i, col in enumerate(columns):
            result = result.with_columns(pl.Series(col, transformed[:, i]))
        return result, imputer

    def transform(
        self,
        df: pl.DataFrame,
        imputer: SimpleImputer,
        columns: list[str],
    ) -> pl.DataFrame:
        """fit 済み imputer を使って df を変換する（データリーク防止）。

        Test データには必ずこのメソッドを使い、Train の統計量を適用する。
        """
        subset = df.select(columns).to_pandas()
        transformed = np.asarray(imputer.transform(subset))

        result = df.clone()
        for i, col in enumerate(columns):
            result = result.with_columns(pl.Series(col, transformed[:, i]))
        return result

    def target_encode(
        self,
        df: pl.DataFrame,
        columns: list[str],
        target_col: str,
        n_splits: int = 5,
        seed: int = 42,
        smoothing: float = 1.0,
    ) -> tuple[pl.DataFrame, TargetEncoder]:
        """OOF CV ベースの Target Encoding を行い、(oof_encoded_df, full_encoder) を返す。

        データリーク防止:
        - val fold の行を encoder の fit に使わない（OOF 実装の核心）
        - full_encoder は全 train で fit した encoder で、Test 時に transform_target_encode() に渡す

        smoothing:
        - 小カテゴリの過学習を防ぐ
        - encoded = (n_i * mean_i + smoothing * global_mean) / (n_i + smoothing)
        - smoothing=0.0 のとき raw mean そのまま

        Args:
            df: Train DataFrame（target_col を含む）
            columns: エンコード対象のカテゴリカラム名リスト
            target_col: ターゲットカラム名
            n_splits: KFold の分割数
            seed: KFold のシード
            smoothing: smoothing 係数（デフォルト 1.0）

        Returns:
            (oof_encoded_df, full_encoder)
            - oof_encoded_df: OOF エンコード済み DataFrame（columns の dtype が Float64）
            - full_encoder: 全 train で fit した encoder
                            Test には transform_target_encode() に渡す
        """
        target = df[target_col].cast(pl.Float64).to_numpy()
        n = len(df)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        result = df.clone()
        full_encoder: TargetEncoder = {}

        for col in columns:
            col_values = df[col].to_list()
            oof_encoded = np.full(n, np.nan, dtype=np.float64)

            # global mean（全 train で計算）
            global_mean = float(np.mean(target))

            # OOF エンコード
            for train_idx, val_idx in kf.split(np.arange(n)):
                train_target = target[train_idx]
                train_col = [col_values[i] for i in train_idx]

                # fold の train で統計量を計算
                fold_global_mean = float(np.mean(train_target))
                cat_stats: dict[str, tuple[float, int]] = {}  # category -> (sum, count)
                for cat, t in zip(train_col, train_target):
                    if cat not in cat_stats:
                        cat_stats[cat] = (0.0, 0)
                    s, c = cat_stats[cat]
                    cat_stats[cat] = (s + float(t), c + 1)

                # val を fold の統計量でエンコード
                for i in val_idx:
                    cat = col_values[i]
                    if cat in cat_stats:
                        s, c = cat_stats[cat]
                        mean_i = s / c
                        encoded = (c * mean_i + smoothing * fold_global_mean) / (c + smoothing)
                    else:
                        encoded = fold_global_mean
                    oof_encoded[i] = encoded

            result = result.with_columns(pl.Series(col, oof_encoded))

            # full_encoder: 全 train で fit
            full_cat_stats: dict[str, tuple[float, int]] = {}
            for cat, t in zip(col_values, target):
                if cat not in full_cat_stats:
                    full_cat_stats[cat] = (0.0, 0)
                s, c = full_cat_stats[cat]
                full_cat_stats[cat] = (s + float(t), c + 1)

            col_encoder: dict[str, float] = {"__global_mean__": global_mean}
            for cat, (s, c) in full_cat_stats.items():
                mean_i = s / c
                col_encoder[cat] = (c * mean_i + smoothing * global_mean) / (c + smoothing)

            full_encoder[col] = col_encoder

        return result, full_encoder

    def transform_target_encode(
        self,
        df: pl.DataFrame,
        encoder: TargetEncoder,
        columns: list[str],
    ) -> pl.DataFrame:
        """full_encoder を使って df（Test データ）を Target Encoding する。

        未知カテゴリは encoder[col]["__global_mean__"] でフォールバックする（KeyError 防止）。

        Args:
            df: Test DataFrame
            encoder: target_encode() が返した full_encoder
            columns: エンコード対象カラム名リスト

        Returns:
            エンコード済み DataFrame（columns の dtype が Float64）
        """
        result = df.clone()
        for col in columns:
            col_encoder = encoder[col]
            global_mean = col_encoder["__global_mean__"]
            encoded = [col_encoder.get(cat, global_mean) for cat in df[col].to_list()]
            result = result.with_columns(pl.Series(col, encoded, dtype=pl.Float64))
        return result
