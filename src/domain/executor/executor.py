"""
Executor Protocol。

PreprocessUseCase は具体的な Executor クラスを知らず、この Protocol に依存する。
"""

from typing import Protocol

from src.domain.data.preprocessor import Node, StepResult
from src.domain.data.table import DataFrame


class Executor(Protocol):
    """前処理パイプラインの実行環境 Protocol。

    UseCase は Executor.run() を呼ぶだけで、
    ローカル/クラウドなどの差異を気にしない。
    """

    def run(
        self,
        nodes: list[Node],
        input_dfs: dict[str, DataFrame],
        targets: list[str],
        output_dir: str,
        cv_splits: list[tuple[list[int], list[int]]] | None,
    ) -> tuple[dict[str, DataFrame], list[StepResult]]:
        """パイプラインを実行して (結果DataFrame map, StepResult list) を返す。"""
        ...
