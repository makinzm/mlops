"""
ExecutorFactory のテスト。

なぜこのテストが必要か:
- local のみ実装済みであり、gcp_vertex / ray_local などの未実装 executor を
  指定した場合は NotImplementedError を上げることを保証する。
- 旧挙動（LocalExecutor へのサイレントフォールバック）は廃止。ユーザーが
  未実装 executor を指定したまま「なぜか local で動いている」状態を防ぐ。
- 未知の executor_type には ValueError を上げ、タイポ等を早期発見できる。
"""

import pytest

from src.infrastructure.executor.factory import ExecutorFactory
from src.infrastructure.executor.local import LocalExecutor


class TestExecutorFactory:
    def test_local_executor_returned_for_local_type(self) -> None:
        """type='local' の場合 LocalExecutor が返ること。"""
        executor = ExecutorFactory.build(executor_type="local")
        assert isinstance(executor, LocalExecutor)

    def test_local_no_fallback(self) -> None:
        """実装済み Executor (local) はフォールバックが False であること。"""
        executor, fallback = ExecutorFactory.build_with_fallback(executor_type="local")
        assert isinstance(executor, LocalExecutor)
        assert fallback is False

    def test_gcp_vertex_raises_not_implemented(self) -> None:
        """gcp_vertex executor は未実装のため NotImplementedError が発生すること。

        旧挙動（LocalExecutor へのサイレントフォールバック）は廃止。
        GCP Vertex AI での学習は usecase=remote_train / vertex_submit を使うこと。
        """
        with pytest.raises(NotImplementedError, match="gcp_vertex"):
            ExecutorFactory.build_with_fallback(executor_type="gcp_vertex")

    def test_ray_local_raises_not_implemented(self) -> None:
        """ray_local executor は未実装のため NotImplementedError が発生すること。"""
        with pytest.raises(NotImplementedError, match="ray_local"):
            ExecutorFactory.build_with_fallback(executor_type="ray_local")

    def test_truly_unknown_raises_value_error(self) -> None:
        """未知の executor_type（タイポ等）には ValueError が発生すること。"""
        with pytest.raises(ValueError, match="typo_executor"):
            ExecutorFactory.build_with_fallback(executor_type="typo_executor")
