"""
PreprocessUseCase — 前処理パイプラインの実行ユースケース。

Hydra Config と注入されたインフラ依存を受け取り、以下を行う:
1. inputs: からデータを読み込む（InputLoaderPort 経由）
2. steps: から Node リストを構築する
3. Executor.run() でパイプラインを実行
4. pipeline_dag.html と preprocess_result.yaml を生成して出力する
5. PreprocessResult を返す
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from src.domain.data.preprocessor import (
    CVSplitterPort,
    InputLoaderPort,
    Node,
    PipelineVisualizerPort,
    PreprocessResult,
    StepResult,
)
from src.domain.data.table import DataFrame
from src.domain.executor.executor import Executor
from src.domain.repository.git import GitRepository
from src.usecase._utils import build_tree_lines as _build_tree_lines


class PreprocessUseCase:
    """前処理パイプラインを実行するユースケース。"""

    def __init__(
        self,
        cfg: DictConfig,
        executor: Executor,
        git_repo: GitRepository,
        input_loader: InputLoaderPort,
        cv_splitter: CVSplitterPort,
        visualizer: PipelineVisualizerPort,
        executor_fallback: bool = False,
        executor_requested: str | None = None,
    ) -> None:
        self._cfg = cfg
        self._executor = executor
        self._git_repo = git_repo
        self._input_loader = input_loader
        self._cv_splitter = cv_splitter
        self._visualizer = visualizer
        self._executor_fallback = executor_fallback
        self._executor_requested = executor_requested

    def execute(self) -> PreprocessResult:
        """パイプラインを実行して PreprocessResult を返す。"""
        cfg = self._cfg
        job_id: str = cfg.get("job_id", datetime.now().strftime("%Y%m%dT%H%M%S"))
        seed: int = int(cfg.get("seed", 42))
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

        # output_dir に .gitignore を生成して parquet 等を git 管理外にする
        self._git_repo.setup_data_dir(Path(str(cfg.output_dir)))

        output_dir = Path(str(cfg.output_dir)) / job_id / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # commit hash の取得（失敗しても続行）
        commit_hash = self._git_repo.get_commit_hash()

        # inputs の読み込み
        input_dfs = self._load_inputs(cfg)

        # steps から Node リストを構築
        nodes = self._build_nodes(cfg, input_dfs)

        # CV splits の生成
        cv_cfg_raw = cfg.get("cv", {})
        cv_cfg_dict: dict[str, object] = (
            dict(OmegaConf.to_container(cv_cfg_raw, resolve=True))  # ty:ignore[no-matching-overload]
            if cv_cfg_raw
            else {}
        )
        cv_splits = self._cv_splitter.build(cv_cfg_dict, input_dfs)

        # targets の取得
        targets_raw = cfg.get("targets", [])
        targets: list[str] = list(OmegaConf.to_container(targets_raw))  # ty:ignore[invalid-argument-type]

        # パイプライン実行
        _, step_results = self._executor.run(
            nodes=nodes,
            input_dfs=input_dfs,
            targets=targets,
            output_dir=str(output_dir),
            cv_splits=cv_splits,
            cv_cfg=cv_cfg_dict,
        )

        # DAG 可視化（output_dir に保存）
        self._visualizer.save_html(nodes, output_dir / "pipeline_dag.html")

        # preprocess_result.yaml の書き出し（output_dir に保存）
        manifest = self._build_manifest(
            job_id=job_id,
            commit_hash=commit_hash,
            step_results=step_results,
            output_dir=output_dir,
        )
        with open(output_dir / "preprocess_result.yaml", "w") as f:
            yaml.dump(manifest, f, allow_unicode=True, default_flow_style=False)

        # README.md の書き出し（ツリー構造）
        self._write_readme(output_dir=output_dir, job_id=job_id, commit_hash=commit_hash)

        return PreprocessResult(
            output_path=output_dir,
            columns=[],
            n_rows=None,
            n_splits=len(cv_splits) if cv_splits else None,
            step_results=step_results,
            commit_hash=commit_hash,
            seed=seed,
            executor_used="local",
            executor_fallback=self._executor_fallback,
            executor_requested=self._executor_requested,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_inputs(self, cfg: DictConfig) -> dict[str, DataFrame]:
        """inputs: 設定から DataFrame を読み込む。"""
        inputs_raw = list(OmegaConf.to_container(cfg.inputs, resolve=True))  # ty:ignore[invalid-argument-type]
        return self._input_loader.load(inputs_raw)

    def _build_nodes(self, cfg: DictConfig, input_dfs: dict[str, DataFrame]) -> list[Node]:
        """inputs: + steps: から Node リストを構築する。"""
        nodes: list[Node] = []

        # Input Nodes
        for inp_id in input_dfs:
            nodes.append(Node(id=inp_id, resolver_cfg={}, is_input=True))

        # Transform Nodes
        steps_raw = OmegaConf.to_container(cfg.steps, resolve=True)
        for step in steps_raw:  # ty:ignore[not-iterable]
            step_dict = dict(step)  # ty:ignore[no-matching-overload]
            node_id = str(step_dict.pop("id", ""))
            from_raw = step_dict.pop("from", None)
            from_nodes: list[str] = []
            if isinstance(from_raw, list):
                from_nodes = [str(f) for f in from_raw]
            elif isinstance(from_raw, str):
                from_nodes = [from_raw]
            resolver_cfg = self._normalize_resolver_cfg(step_dict)
            nodes.append(Node(id=node_id, resolver_cfg=resolver_cfg, from_nodes=from_nodes))

        return nodes

    def _normalize_resolver_cfg(self, step_dict: dict[str, Any]) -> dict[str, Any]:
        """step の辞書から resolver_cfg 形式に正規化する。"""
        result: dict[str, Any] = {}
        for resolver_name, resolver_cfg in step_dict.items():
            cfg_dict = dict(resolver_cfg) if isinstance(resolver_cfg, dict) else {}
            if resolver_name == "output" and "method" not in cfg_dict:
                cfg_dict["method"] = "output"
            result[resolver_name] = cfg_dict
        return result

    def _write_readme(self, output_dir: Path, job_id: str, commit_hash: str) -> None:
        """output_dir に README.md を生成する。ファイルツリーを含む。"""
        lines = [
            f"# Preprocess Result — `{job_id}`",
            "",
            f"- **commit**: `{commit_hash}`",
            f"- **output_dir**: `{output_dir}`",
            "",
            "## Output Tree",
            "",
            "```",
            output_dir.name + "/",
        ]
        lines += _build_tree_lines(output_dir)
        lines += ["```", ""]
        (output_dir / "README.md").write_text("\n".join(lines))

    def _build_manifest(
        self,
        job_id: str,
        commit_hash: str,
        step_results: list[StepResult],
        output_dir: Path,
    ) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "job_id": job_id,
            "timestamp": datetime.now().isoformat(),
            "commit_hash": commit_hash,
            "executor_used": "local",
            "dag_path": "pipeline_dag.html",
            "output_dir": str(output_dir),
            "step_results": [
                {
                    "resolver": r.resolver,
                    "method": r.method,
                    "status": r.status,
                    "reason": r.reason,
                }
                for r in step_results
            ],
        }
        if self._executor_fallback:
            manifest["executor_fallback"] = True
            manifest["executor_requested"] = self._executor_requested
        return manifest
