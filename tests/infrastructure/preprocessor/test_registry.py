"""
RESOLVER_REGISTRY の graceful skip テスト。

なぜこのテストが必要か:
- 未登録 Resolver や未実装 Method への呼び出しが例外を上げず、
  StepResult(status="skipped") を返すことがパイプライン継続の根幹。
- このテストが失敗すると、1ステップの問題で全体のパイプラインが止まる。
- 実行中のエラー（failed）でも後続ステップが継続されることを確認する。
"""

import polars as pl
import pytest

from src.infrastructure.preprocessor.registry import run_step


@pytest.fixture()
def sample_df() -> pl.DataFrame:
    return pl.DataFrame({"id": [1, 2, 3], "val": [1.0, 2.0, 3.0]})


class TestRunStep:
    def test_resolver_not_found_returns_skipped(self, sample_df: pl.DataFrame) -> None:
        """未登録 Resolver は StepResult(status='skipped') を返すこと。"""
        result_df, step_result = run_step(
            df=sample_df,
            resolver_name="torchvision",
            method="embed_image",
            kwargs={},
        )
        assert result_df is sample_df  # DataFrame は変わらない
        assert step_result.status == "skipped"
        assert "resolver not found" in (step_result.reason or "")

    def test_method_not_found_returns_skipped(self, sample_df: pl.DataFrame) -> None:
        """登録済み Resolver に存在しない Method は StepResult(status='skipped') を返すこと。"""
        result_df, step_result = run_step(
            df=sample_df,
            resolver_name="polars",
            method="nonexistent_method",
            kwargs={},
        )
        assert result_df is sample_df
        assert step_result.status == "skipped"
        assert "method not found" in (step_result.reason or "")

    def test_ok_step_returns_transformed_df(self, sample_df: pl.DataFrame) -> None:
        """正常なステップは StepResult(status='ok') と変換後 DataFrame を返すこと。"""
        result_df, step_result = run_step(
            df=sample_df,
            resolver_name="polars",
            method="select_columns",
            kwargs={"columns": ["id"]},
        )
        assert step_result.status == "ok"
        assert result_df.columns == ["id"]

    def test_execution_error_returns_failed_and_continues(self, sample_df: pl.DataFrame) -> None:
        """実行中に例外が発生しても StepResult(status='failed') を返し、例外を上げないこと。

        後続ステップが継続できるよう、DataFrame は変更前のものを返す。
        """
        # 存在しないカラムを select しようとしてエラーを起こす
        result_df, step_result = run_step(
            df=sample_df,
            resolver_name="polars",
            method="select_columns",
            kwargs={"columns": ["nonexistent_column"]},
        )
        assert result_df is sample_df  # 変換前 DataFrame が返る
        assert step_result.status == "failed"
        assert step_result.reason is not None

    def test_registered_resolvers_include_polars_and_sklearn(self) -> None:
        """RESOLVER_REGISTRY に polars / sklearn が登録されていること。"""
        from src.infrastructure.preprocessor.registry import RESOLVER_REGISTRY

        assert "polars" in RESOLVER_REGISTRY
        assert "sklearn" in RESOLVER_REGISTRY
        assert "output" in RESOLVER_REGISTRY
