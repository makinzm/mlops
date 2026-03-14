"""
SklearnResolver の fill_na メソッドに対するユニットテスト。

なぜこのテストが必要か:
- fill_na は Train データで fit した統計量（median/mean/constant）を
  Test データに transform することで、データリークを防ぐ設計になっている。
- 特に「Test データの median ではなく Train の median で埋める」ことを
  明示的に検証する必要がある。これがないと将来の変更でリークが生じても検知できない。
- sklearn の SimpleImputer の fit/transform 分離を wrapper として正しく実装できているかを確認する。
"""

import polars as pl
import pytest

from src.infrastructure.preprocessor.resolvers.sklearn_resolver import SklearnResolver


@pytest.fixture()
def resolver() -> SklearnResolver:
    return SklearnResolver()


class TestFillNaMedian:
    def test_fill_na_fills_nulls(self, resolver: SklearnResolver) -> None:
        """median 戦略で null が埋められること。"""
        train_df = pl.DataFrame({"col1": [1.0, 2.0, None, 4.0, 5.0]})
        result_train, _ = resolver.fill_na(train_df, strategy="median", columns=["col1"])
        assert result_train["col1"].null_count() == 0

    def test_fill_na_median_value(self, resolver: SklearnResolver) -> None:
        """null が Train データの median（3.0）で埋められること。"""
        train_df = pl.DataFrame({"col1": [1.0, 2.0, None, 4.0, 5.0]})
        result_train, _ = resolver.fill_na(train_df, strategy="median", columns=["col1"])
        # null だった箇所が median=3.0 で埋まること（null が index=2 にある）
        # 全体の median は [1,2,4,5] の中央値 = (2+4)/2 = 3.0
        assert result_train["col1"][2] == 3.0

    def test_fill_na_no_data_leak(self, resolver: SklearnResolver) -> None:
        """Train の統計量が Test に適用されること（データリーク防止）。

        Train: [1, 2, 3] → median = 2.0
        Test:  [10, None, 30]
        → Test の null は Train の median (2.0) で埋まるべき。
           Test 自体の median (20.0) で埋まってはいけない。
        """
        train_df = pl.DataFrame({"col1": [1.0, 2.0, 3.0]})
        test_df = pl.DataFrame({"col1": [10.0, None, 30.0]})
        _, imputer = resolver.fill_na(train_df, strategy="median", columns=["col1"])
        result_test = resolver.transform(test_df, imputer=imputer, columns=["col1"])
        # Train median は 2.0、Test 単体の median は 20.0
        assert result_test["col1"][1] == 2.0

    def test_fill_na_mean_strategy(self, resolver: SklearnResolver) -> None:
        """mean 戦略で null が Train の mean で埋められること。"""
        train_df = pl.DataFrame({"col1": [1.0, 2.0, None, 4.0]})
        result_train, _ = resolver.fill_na(train_df, strategy="mean", columns=["col1"])
        # mean of [1, 2, 4] = 7/3 ≈ 2.333...
        filled_val = result_train["col1"][2]
        assert filled_val is not None
        assert abs(filled_val - (1.0 + 2.0 + 4.0) / 3.0) < 1e-6

    def test_fill_na_constant_strategy(self, resolver: SklearnResolver) -> None:
        """constant 戦略で null が指定値（-999）で埋められること。"""
        train_df = pl.DataFrame({"col1": [1.0, None, 3.0]})
        result_train, _ = resolver.fill_na(
            train_df, strategy="constant", columns=["col1"], fill_value=-999.0
        )
        assert result_train["col1"][1] == -999.0

    def test_fill_na_multiple_columns(self, resolver: SklearnResolver) -> None:
        """複数カラムを同時に補完できること。"""
        train_df = pl.DataFrame(
            {
                "col1": [1.0, None, 3.0],
                "col2": [None, 2.0, 3.0],
                "label": [0, 1, 0],
            }
        )
        result_train, _ = resolver.fill_na(
            train_df, strategy="median", columns=["col1", "col2"]
        )
        assert result_train["col1"].null_count() == 0
        assert result_train["col2"].null_count() == 0
        # label カラムは変更されない
        assert result_train["label"].to_list() == [0, 1, 0]

    def test_supported_methods_includes_fill_na(self, resolver: SklearnResolver) -> None:
        """supported_methods() に fill_na が含まれること。"""
        assert "fill_na" in resolver.supported_methods()
