"""
sklearn ベースの変換 Resolver。

対応メソッド:
- fill_na : 欠損補完（median/mean/constant）
            Train で fit した統計量を Test に transform することでデータリークを防ぐ。

設計上の注意:
- fill_na は (transformed_df, fitted_imputer) のタプルを返す。
  DAGRunner は fitted_imputer を保持し、Test データには transform() を呼ぶ。
- Polars の null を pandas 経由で SimpleImputer に渡し、結果を Polars に戻す。
"""

from typing import Any

import polars as pl
from sklearn.impute import SimpleImputer


class SklearnResolver:
    """sklearn を使ったタブラーデータ変換 Resolver。"""

    def supported_methods(self) -> set[str]:
        return {"fill_na"}

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
            result, _ = self.fill_na(df, strategy=strategy, columns=columns, fill_value=fill_value)
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
        transformed = imputer.transform(subset)

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
        transformed = imputer.transform(subset)

        result = df.clone()
        for i, col in enumerate(columns):
            result = result.with_columns(pl.Series(col, transformed[:, i]))
        return result
