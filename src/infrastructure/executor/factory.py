"""
ExecutorFactory — executor_type 文字列から Executor インスタンスを生成する。

未実装 Executor が指定された場合はサイレントフォールバックせず NotImplementedError を上げる。
既知だが未実装の型（gcp_vertex / ray_local）と、全く未知の型（タイポ等）は別の例外で区別する。
"""

from src.infrastructure.executor.local import LocalExecutor

# 既知だが未実装の Executor とエラーメッセージ
_UNIMPLEMENTED_EXECUTORS: dict[str, str] = {
    "gcp_vertex": (
        "gcp_vertex executor は preprocess ステップでは未実装です。"
        " GCP Vertex AI 学習は usecase=remote_train / vertex_submit を使ってください。"
    ),
    "ray_local": (
        "ray_local executor は未実装です。 ローカル実行には executor: local を使用してください。"
    ),
}


class ExecutorFactory:
    """Executor のファクトリ。"""

    @staticmethod
    def build(executor_type: str) -> LocalExecutor:
        """executor_type に対応する Executor を返す。

        未実装 / 未知の場合は例外を上げる。
        フォールバック有無を知りたい場合は build_with_fallback() を使う。
        """
        executor, _ = ExecutorFactory.build_with_fallback(executor_type)
        return executor

    @staticmethod
    def build_with_fallback(executor_type: str) -> tuple[LocalExecutor, bool]:
        """executor_type に対応する Executor と fallback フラグを返す。

        Returns:
            (executor, is_fallback=False) — local のみ実装済みのため常に False。

        Raises:
            NotImplementedError: 既知だが未実装の executor_type（gcp_vertex, ray_local）。
            ValueError: 未知の executor_type（タイポ等）。
        """
        if executor_type == "local":
            return LocalExecutor(), False
        if executor_type in _UNIMPLEMENTED_EXECUTORS:
            raise NotImplementedError(_UNIMPLEMENTED_EXECUTORS[executor_type])
        raise ValueError(
            f"未知の executor_type: {executor_type!r}。"
            f" 利用可能: ['local']、未実装: {sorted(_UNIMPLEMENTED_EXECUTORS)}"
        )
