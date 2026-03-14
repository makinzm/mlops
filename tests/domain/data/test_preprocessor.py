"""
domain/data/preprocessor.py のデータクラスに対するユニットテスト。

なぜこのテストが必要か:
- StepResult / ColumnMeta / PreprocessResult / Node は後続の全レイヤーが依存するコアデータ構造。
- フィールド名・型・デフォルト値を固定することで、Resolver や UseCase からの参照が安全になる。
- status の "ok" / "skipped" / "failed" の3値を明示的に検証し、
  後続テストで期待する文字列を固定する。
"""

from pathlib import Path

import pytest

from src.domain.data.preprocessor import (
    ColumnMeta,
    Node,
    PreprocessResult,
    StepResult,
)


class TestStepResult:
    def test_ok(self) -> None:
        """status='ok' の StepResult が生成できること。"""
        result = StepResult(resolver="polars", method="select_columns", status="ok", reason=None)
        assert result.resolver == "polars"
        assert result.method == "select_columns"
        assert result.status == "ok"
        assert result.reason is None

    def test_skipped_with_reason(self) -> None:
        """status='skipped' + reason が格納されること。"""
        result = StepResult(
            resolver="torchvision",
            method="embed_image",
            status="skipped",
            reason="resolver not found",
        )
        assert result.status == "skipped"
        assert result.reason == "resolver not found"

    def test_failed_with_reason(self) -> None:
        """status='failed' + reason が格納されること。"""
        result = StepResult(
            resolver="polars",
            method="arithmetic",
            status="failed",
            reason="ZeroDivisionError: division by zero",
        )
        assert result.status == "failed"
        assert "ZeroDivisionError" in (result.reason or "")


class TestColumnMeta:
    def test_fields(self) -> None:
        """ColumnMeta に name / modality / dtype が格納されること。"""
        meta = ColumnMeta(name="price", modality="tabular", dtype="float32")
        assert meta.name == "price"
        assert meta.modality == "tabular"
        assert meta.dtype == "float32"

    def test_image_embed_modality(self) -> None:
        """画像 embedding カラムのモダリティが 'image_embed' になること。"""
        meta = ColumnMeta(name="image_embed", modality="image_embed", dtype="List[float32]")
        assert meta.modality == "image_embed"


class TestPreprocessResult:
    def test_all_fields(self) -> None:
        """PreprocessResult の全フィールドが設定・参照できること。"""
        step_results = [
            StepResult(resolver="polars", method="select_columns", status="ok", reason=None),
        ]
        columns = [ColumnMeta(name="price", modality="tabular", dtype="float32")]
        result = PreprocessResult(
            output_path=Path("data/processed/my_job/2026-03-15/"),
            columns=columns,
            n_rows=1000,
            n_splits=5,
            step_results=step_results,
            commit_hash="abc1234",
            seed=42,
        )
        assert result.output_path == Path("data/processed/my_job/2026-03-15/")
        assert result.n_rows == 1000
        assert result.n_splits == 5
        assert result.seed == 42
        assert result.commit_hash == "abc1234"
        assert len(result.step_results) == 1
        assert len(result.columns) == 1

    def test_n_splits_none_when_no_cv(self) -> None:
        """CV なしの場合 n_splits が None になること。"""
        result = PreprocessResult(
            output_path=Path("data/processed/"),
            columns=[],
            n_rows=None,
            n_splits=None,
            step_results=[],
            commit_hash="abc1234",
            seed=42,
        )
        assert result.n_splits is None


class TestNode:
    def test_minimal_node(self) -> None:
        """from_nodes を省略した Node が生成できること（from_nodes はデフォルト空リスト）。"""
        node = Node(id="selected", resolver_cfg={"polars": {"method": "select_columns"}})
        assert node.id == "selected"
        assert node.from_nodes == []

    def test_explicit_from_nodes(self) -> None:
        """from_nodes に明示的な依存 Node id リストを設定できること。"""
        node = Node(
            id="merged",
            from_nodes=["with_calc", "raw_images"],
            resolver_cfg={"polars": {"method": "join"}},
        )
        assert node.from_nodes == ["with_calc", "raw_images"]

    def test_input_node(self) -> None:
        """Input Node（is_input=True）を識別できること。"""
        node = Node(
            id="raw_train",
            from_nodes=[],
            resolver_cfg={},
            is_input=True,
        )
        assert node.is_input is True

    def test_transform_node_is_not_input(self) -> None:
        """通常の変換 Node は is_input=False であること。"""
        node = Node(id="selected", resolver_cfg={"polars": {"method": "select_columns"}})
        assert node.is_input is False


class TestPreprocessResultExecutorFallback:
    def test_executor_fallback_recorded(self) -> None:
        """executor_fallback=True が設定できること（未実装 Executor へのフォールバック時）。"""
        result = PreprocessResult(
            output_path=Path("data/processed/"),
            columns=[],
            n_rows=None,
            n_splits=None,
            step_results=[],
            commit_hash="abc1234",
            seed=42,
            executor_used="local",
            executor_fallback=True,
            executor_requested="gcp_vertex",
        )
        assert result.executor_fallback is True
        assert result.executor_requested == "gcp_vertex"
        assert result.executor_used == "local"

    def test_executor_fallback_default_false(self) -> None:
        """executor_fallback のデフォルトは False であること。"""
        result = PreprocessResult(
            output_path=Path("data/processed/"),
            columns=[],
            n_rows=None,
            n_splits=None,
            step_results=[],
            commit_hash="abc1234",
            seed=42,
        )
        assert result.executor_fallback is False
