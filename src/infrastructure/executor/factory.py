"""
ExecutorFactory — executor_type 文字列から Executor インスタンスを生成する。

未実装 Executor が指定された場合は LocalExecutor にフォールバックし、
呼び出し元が fallback フラグを確認できるよう build_with_fallback() を提供する。
"""

from src.infrastructure.executor.local import LocalExecutor

# 実装済み Executor のキー一覧
_IMPLEMENTED_EXECUTORS: set[str] = {"local"}


class ExecutorFactory:
    """Executor のファクトリ。"""

    @staticmethod
    def build(executor_type: str) -> LocalExecutor:
        """executor_type に対応する Executor を返す。

        未実装の場合は LocalExecutor を返す（例外は上げない）。
        フォールバック有無を知りたい場合は build_with_fallback() を使う。
        """
        executor, _ = ExecutorFactory.build_with_fallback(executor_type)
        return executor

    @staticmethod
    def build_with_fallback(executor_type: str) -> tuple[LocalExecutor, bool]:
        """executor_type に対応する Executor と fallback フラグを返す。

        Returns:
            (executor, is_fallback)
            is_fallback=True の場合、指定 executor は未実装で local に落とした。
        """
        if executor_type == "local":
            return LocalExecutor(), False
        # 将来: ray_local / gcp_vertex / etc. はここに追加
        # 未実装の場合は local にフォールバック
        return LocalExecutor(), True
