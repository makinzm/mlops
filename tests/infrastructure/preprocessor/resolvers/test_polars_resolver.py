"""
PolarsResolver の各メソッドに対するユニットテスト。

なぜこのテストが必要か:
- polars:select_columns / arithmetic / exp_weight / join は
  パイプラインの主要変換処理であり、戻り値の形状・カラム名・数値を
  明示的に検証して回帰を防ぐ必要がある。
- join は複数 Node の DataFrame をマージするため、
  DAGRunner から呼ばれる際の入力フォーマット（DataFrameのリスト）を想定した設計にする。
- exp_weight は「時系列の新しいデータほど重みが高い」挙動を検証する。
"""

import polars as pl
import pytest

from src.infrastructure.preprocessor.resolvers.polars_resolver import PolarsResolver


@pytest.fixture()
def resolver() -> PolarsResolver:
    return PolarsResolver()


@pytest.fixture()
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "label": [0, 1, 0, 1, 0],
            "extra": ["a", "b", "c", "d", "e"],
        }
    )


class TestSelectColumns:
    def test_select_keeps_specified_columns(
        self, resolver: PolarsResolver, sample_df: pl.DataFrame
    ) -> None:
        """指定カラムのみが残ること。"""
        result = resolver.select_columns(sample_df, columns=["id", "col1", "label"])
        assert result.columns == ["id", "col1", "label"]
        assert len(result) == 5

    def test_select_removes_unspecified_columns(
        self, resolver: PolarsResolver, sample_df: pl.DataFrame
    ) -> None:
        """未指定カラム (extra, col2) が除去されること。"""
        result = resolver.select_columns(sample_df, columns=["id", "col1", "label"])
        assert "extra" not in result.columns
        assert "col2" not in result.columns

    def test_supported_methods_includes_select_columns(self, resolver: PolarsResolver) -> None:
        """supported_methods() に select_columns が含まれること。"""
        assert "select_columns" in resolver.supported_methods()


class TestArithmetic:
    def test_add(self, resolver: PolarsResolver, sample_df: pl.DataFrame) -> None:
        """add 演算が col_a + col_b を output_col に追加すること。"""
        result = resolver.arithmetic(
            sample_df, operation="add", col_a="col1", col_b="col2", output_col="sum_col"
        )
        assert "sum_col" in result.columns
        assert result["sum_col"].to_list() == [11.0, 22.0, 33.0, 44.0, 55.0]

    def test_multiply(self, resolver: PolarsResolver, sample_df: pl.DataFrame) -> None:
        """multiply 演算が col_a * col_b を output_col に追加すること。"""
        result = resolver.arithmetic(
            sample_df, operation="multiply", col_a="col1", col_b="col2", output_col="prod_col"
        )
        assert result["prod_col"].to_list() == [10.0, 40.0, 90.0, 160.0, 250.0]

    def test_log1p_single_column(self, resolver: PolarsResolver, sample_df: pl.DataFrame) -> None:
        """log1p 演算が col_a の log1p を output_col に追加すること（col_b は不要）。"""
        import math

        result = resolver.arithmetic(
            sample_df, operation="log1p", col_a="col1", output_col="log_col"
        )
        assert "log_col" in result.columns
        assert abs(result["log_col"][0] - math.log1p(1.0)) < 1e-6

    def test_subtract(self, resolver: PolarsResolver, sample_df: pl.DataFrame) -> None:
        """subtract 演算が col_a - col_b を output_col に追加すること。"""
        result = resolver.arithmetic(
            sample_df, operation="subtract", col_a="col2", col_b="col1", output_col="diff_col"
        )
        assert result["diff_col"].to_list() == [9.0, 18.0, 27.0, 36.0, 45.0]

    def test_divide(self, resolver: PolarsResolver, sample_df: pl.DataFrame) -> None:
        """divide 演算が col_a / col_b を output_col に追加すること。"""
        result = resolver.arithmetic(
            sample_df, operation="divide", col_a="col2", col_b="col1", output_col="ratio_col"
        )
        assert result["ratio_col"].to_list() == [10.0, 10.0, 10.0, 10.0, 10.0]


class TestExpWeight:
    def test_weight_column_added(self, resolver: PolarsResolver) -> None:
        """__weight__ カラムが追加されること。"""
        df = pl.DataFrame({"id": [1, 2, 3], "date": ["2026-01-01", "2026-01-02", "2026-01-03"]})
        result = resolver.exp_weight(df, time_col="date", decay=0.95, weight_col="__weight__")
        assert "__weight__" in result.columns

    def test_newer_rows_have_higher_weight(self, resolver: PolarsResolver) -> None:
        """時系列で新しい行ほど重みが高いこと。"""
        df = pl.DataFrame({"id": [1, 2, 3], "date": ["2026-01-01", "2026-01-02", "2026-01-03"]})
        result = resolver.exp_weight(df, time_col="date", decay=0.95, weight_col="__weight__")
        weights = result["__weight__"].to_list()
        # 昇順ソートで最後（最新）が最大重み
        assert weights[0] < weights[1] < weights[2]

    def test_weight_col_name_respected(self, resolver: PolarsResolver) -> None:
        """weight_col に指定した名前でカラムが追加されること。"""
        df = pl.DataFrame({"id": [1, 2], "date": ["2026-01-01", "2026-01-02"]})
        result = resolver.exp_weight(df, time_col="date", decay=0.9, weight_col="my_weight")
        assert "my_weight" in result.columns


class TestJoin:
    def test_left_join_on_id(self, resolver: PolarsResolver) -> None:
        """left join で左 DataFrame の行が全て保持されること。"""
        left = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        right = pl.DataFrame({"id": [1, 2], "extra": ["a", "b"]})
        result = resolver.join([left, right], on="id", how="left")
        assert len(result) == 3
        assert "extra" in result.columns

    def test_inner_join_on_id(self, resolver: PolarsResolver) -> None:
        """inner join でマッチする行のみ残ること。"""
        left = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        right = pl.DataFrame({"id": [1, 2], "extra": ["a", "b"]})
        result = resolver.join([left, right], on="id", how="inner")
        assert len(result) == 2
