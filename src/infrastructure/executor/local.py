"""
LocalExecutor — ローカルプロセスで逐次実行する Executor。

デフォルトの実行環境。開発・小規模データ向け。
"""

from pathlib import Path

from src.domain.data.preprocessor import Node, StepResult
from src.domain.data.table import DataFrame
from src.infrastructure.preprocessor.dag_runner import DAGRunner


class LocalExecutor:
    """ローカルプロセスで DAGRunner を直接実行する Executor。"""

    def run(
        self,
        nodes: list[Node],
        input_dfs: dict[str, DataFrame],
        targets: list[str],
        output_dir: str,
        cv_splits: list[tuple[list[int], list[int]]] | None,
        cv_cfg: dict[str, object] | None = None,
    ) -> tuple[dict[str, DataFrame], list[StepResult]]:
        """DAGRunner を使ってパイプラインをローカル実行する。"""
        runner = DAGRunner(
            nodes=nodes,
            input_dfs=input_dfs,
            output_dir=Path(output_dir),
            cv_splits=cv_splits,
            cv_cfg=cv_cfg,
        )
        results = runner.run(targets=targets)
        return results, runner.get_step_results()
