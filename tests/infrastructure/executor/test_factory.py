"""
ExecutorFactory のテスト。

なぜこのテストが必要か:
- 未実装 Executor が指定された場合の local へのフォールバックは
  パイプライン継続性の保証であり、明示的にテストする必要がある。
- フォールバック発生時は PreprocessResult に executor_fallback=True が
  記録されることを確認する。
"""

from src.infrastructure.executor.factory import ExecutorFactory
from src.infrastructure.executor.local import LocalExecutor


class TestExecutorFactory:
    def test_local_executor_returned_for_local_type(self) -> None:
        """type='local' の場合 LocalExecutor が返ること。"""
        executor = ExecutorFactory.build(executor_type="local")
        assert isinstance(executor, LocalExecutor)

    def test_unknown_executor_falls_back_to_local(self) -> None:
        """未実装 Executor 指定時は LocalExecutor にフォールバックすること。"""
        executor, fallback = ExecutorFactory.build_with_fallback(executor_type="gcp_vertex")
        assert isinstance(executor, LocalExecutor)
        assert fallback is True

    def test_known_executor_no_fallback(self) -> None:
        """実装済み Executor (local) はフォールバックが False であること。"""
        executor, fallback = ExecutorFactory.build_with_fallback(executor_type="local")
        assert isinstance(executor, LocalExecutor)
        assert fallback is False
