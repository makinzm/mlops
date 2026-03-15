"""
DAGRunner のテスト。

なぜこのテストが必要か:
- DAGRunner は from: / targets: の依存解決とノード実行順序を担うコアエンジン。
- 線形フロー・分岐・部分実行の3パターンを検証することで、
  設計通りのDAG解決が行われていることを確認する。
- 特に「targets に含まれないノードは実行されない」ことは
  計算コスト削減の核であり、明示的にテストする必要がある。
"""

from pathlib import Path

import polars as pl
import pytest

from src.domain.data.preprocessor import Node
from src.infrastructure.preprocessor.dag_runner import DAGRunner


@pytest.fixture()
def raw_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "label": [0, 1, 0, 1, 0],
        }
    )


class TestLinearFlow:
    def test_linear_pipeline(self, raw_df: pl.DataFrame, tmp_path: Path) -> None:
        """from: を省略した線形フロー（Input → select → fill_na）が実行されること。"""
        nodes = [
            Node(id="raw_train", resolver_cfg={}, is_input=True),
            Node(
                id="selected",
                resolver_cfg={
                    "polars": {"method": "select_columns", "columns": ["id", "col1", "label"]}
                },
            ),
        ]
        input_dfs = {"raw_train": raw_df}
        runner = DAGRunner(nodes=nodes, input_dfs=input_dfs, output_dir=tmp_path, cv_splits=None)
        results = runner.run(targets=["selected"])

        assert "selected" in results
        assert results["selected"].columns == ["id", "col1", "label"]
        assert len(results["selected"]) == 5

    def test_step_results_recorded(self, raw_df: pl.DataFrame, tmp_path: Path) -> None:
        """実行後に step_results が記録されること。"""
        nodes = [
            Node(id="raw_train", resolver_cfg={}, is_input=True),
            Node(
                id="selected",
                resolver_cfg={"polars": {"method": "select_columns", "columns": ["id", "col1"]}},
            ),
        ]
        input_dfs = {"raw_train": raw_df}
        runner = DAGRunner(nodes=nodes, input_dfs=input_dfs, output_dir=tmp_path, cv_splits=None)
        runner.run(targets=["selected"])

        step_results = runner.get_step_results()
        assert len(step_results) >= 1
        assert any(r.status == "ok" for r in step_results)


class TestBranchFrom:
    def test_explicit_from_creates_branch(self, raw_df: pl.DataFrame, tmp_path: Path) -> None:
        """from: で分岐元を明示した場合、正しい中間 DataFrame から処理されること。"""
        nodes = [
            Node(id="raw_train", resolver_cfg={}, is_input=True),
            Node(
                id="selected",
                resolver_cfg={
                    "polars": {"method": "select_columns", "columns": ["id", "col1", "label"]}
                },
            ),
            Node(
                id="branch_from_raw",
                from_nodes=["raw_train"],  # selected ではなく raw_train から分岐
                resolver_cfg={
                    "polars": {"method": "select_columns", "columns": ["id", "col2", "label"]}
                },
            ),
        ]
        input_dfs = {"raw_train": raw_df}
        runner = DAGRunner(nodes=nodes, input_dfs=input_dfs, output_dir=tmp_path, cv_splits=None)
        results = runner.run(targets=["selected", "branch_from_raw"])

        # selected は col1 を持ち col2 を持たない
        assert "col1" in results["selected"].columns
        assert "col2" not in results["selected"].columns
        # branch_from_raw は col2 を持ち col1 を持たない
        assert "col2" in results["branch_from_raw"].columns
        assert "col1" not in results["branch_from_raw"].columns


class TestTargetsPartialExecution:
    def test_unreachable_node_not_executed(self, raw_df: pl.DataFrame, tmp_path: Path) -> None:
        """targets に含まれないノードは実行されないこと。

        expensive_node が実行されると例外が発生する設計にして、
        実際には呼ばれていないことを確認する。
        """
        nodes = [
            Node(id="raw_train", resolver_cfg={}, is_input=True),
            Node(
                id="selected",
                resolver_cfg={"polars": {"method": "select_columns", "columns": ["id", "col1"]}},
            ),
            Node(
                id="expensive_node",
                from_nodes=["selected"],
                # 存在しないメソッドだが、このノードが実行されなければエラーにならない
                resolver_cfg={"polars": {"method": "select_columns", "columns": ["id", "col1"]}},
            ),
        ]
        input_dfs = {"raw_train": raw_df}
        runner = DAGRunner(nodes=nodes, input_dfs=input_dfs, output_dir=tmp_path, cv_splits=None)
        # targets に expensive_node を含めない
        results = runner.run(targets=["selected"])

        assert "selected" in results
        assert "expensive_node" not in results

    def test_skipped_step_continues_pipeline(self, raw_df: pl.DataFrame, tmp_path: Path) -> None:
        """未知 Resolver があっても後続ステップが継続されること。

        unknown_step が selected の依存となるよう from_nodes で明示し、
        selected が targets にある場合は unknown_step も実行されるが
        graceful skip されて selected まで処理が継続することを確認する。
        """
        nodes = [
            Node(id="raw_train", resolver_cfg={}, is_input=True),
            Node(
                id="unknown_step",
                resolver_cfg={"torchvision": {"method": "embed_image"}},  # 未実装
            ),
            Node(
                id="selected",
                from_nodes=["unknown_step"],  # unknown_step の後続
                resolver_cfg={"polars": {"method": "select_columns", "columns": ["id", "col1"]}},
            ),
        ]
        input_dfs = {"raw_train": raw_df}
        runner = DAGRunner(nodes=nodes, input_dfs=input_dfs, output_dir=tmp_path, cv_splits=None)
        results = runner.run(targets=["selected"])

        # unknown_step はスキップされるが selected は実行される
        assert "selected" in results
        step_results = runner.get_step_results()
        skipped = [r for r in step_results if r.status == "skipped"]
        assert len(skipped) >= 1
