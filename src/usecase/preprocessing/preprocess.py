"""
PreprocessUseCase — 前処理パイプラインの実行ユースケース。

Hydra Config を受け取り、以下を行う:
1. inputs: からデータを読み込む
2. steps: から Node リストを構築する
3. ExecutorFactory で Executor を生成（フォールバックを検出）
4. Executor.run() でパイプラインを実行
5. pipeline_dag.html と preprocess_result.yaml を生成して出力する
6. PreprocessResult を返す
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from omegaconf import DictConfig, OmegaConf

from src.domain.data.preprocessor import Node, PreprocessResult, StepResult
from src.infrastructure.executor.factory import ExecutorFactory
from src.infrastructure.preprocessor.visualizer import PipelineVisualizer
from src.infrastructure.repository.git import GitRepositoryImpl


class PreprocessUseCase:
    """前処理パイプラインを実行するユースケース。"""

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    def execute(self) -> PreprocessResult:
        """パイプラインを実行して PreprocessResult を返す。"""
        cfg = self._cfg
        job_id: str = cfg.get("job_id", datetime.now().strftime("%Y%m%dT%H%M%S"))
        seed: int = int(cfg.get("seed", 42))
        output_dir = Path(str(cfg.output_dir)) / job_id / datetime.now().strftime("%Y%m%dT%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        # commit hash の取得（失敗しても続行）
        commit_hash = self._get_commit_hash()

        # Executor の生成（フォールバック検出）
        executor_type: str = str(cfg.executor.type)
        executor, is_fallback = ExecutorFactory.build_with_fallback(executor_type)

        # inputs の読み込み
        input_dfs = self._load_inputs(cfg)

        # steps から Node リストを構築
        nodes = self._build_nodes(cfg, input_dfs)

        # CV splits の生成
        cv_splits = self._build_cv_splits(cfg, input_dfs)

        # targets の取得
        targets_raw = cfg.get("targets", [])
        targets: list[str] = list(OmegaConf.to_container(targets_raw))  # type: ignore[arg-type]

        # パイプライン実行
        _, step_results = executor.run(
            nodes=nodes,
            input_dfs=input_dfs,
            targets=targets,
            output_dir=str(output_dir),
            cv_splits=cv_splits,
        )

        # DAG 可視化
        PipelineVisualizer(nodes).save_html(output_dir / "pipeline_dag.html")

        # preprocess_result.yaml の書き出し
        manifest = self._build_manifest(
            job_id=job_id,
            commit_hash=commit_hash,
            executor_type=executor_type,
            is_fallback=is_fallback,
            step_results=step_results,
            output_dir=output_dir,
        )
        with open(output_dir / "preprocess_result.yaml", "w") as f:
            yaml.dump(manifest, f, allow_unicode=True, default_flow_style=False)

        return PreprocessResult(
            output_path=output_dir,
            columns=[],
            n_rows=None,
            n_splits=len(cv_splits) if cv_splits else None,
            step_results=step_results,
            commit_hash=commit_hash,
            seed=seed,
            executor_used="local",
            executor_fallback=is_fallback,
            executor_requested=executor_type if is_fallback else None,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_commit_hash(self) -> str:
        """GitRepositoryImpl 経由でコミットハッシュを取得する。"""
        return GitRepositoryImpl().get_commit_hash()

    def _load_inputs(self, cfg: DictConfig) -> dict[str, pl.DataFrame]:
        """inputs: 設定から DataFrame を読み込む。"""
        input_dfs: dict[str, pl.DataFrame] = {}
        inputs_raw = OmegaConf.to_container(cfg.inputs, resolve=True)
        for inp in inputs_raw:  # type: ignore[union-attr]
            inp_dict = dict(inp)  # type: ignore[arg-type]
            inp_id = str(inp_dict["id"])
            fmt = str(inp_dict.get("format", "csv"))
            if "path" in inp_dict:
                path = Path(str(inp_dict["path"]))
                if fmt == "csv":
                    input_dfs[inp_id] = pl.read_csv(path)
                elif fmt == "parquet":
                    input_dfs[inp_id] = pl.read_parquet(path)
                else:
                    input_dfs[inp_id] = pl.read_csv(path)
        return input_dfs

    def _build_nodes(self, cfg: DictConfig, input_dfs: dict[str, pl.DataFrame]) -> list[Node]:
        """inputs: + steps: から Node リストを構築する。"""
        nodes: list[Node] = []

        # Input Nodes
        for inp_id in input_dfs:
            nodes.append(Node(id=inp_id, resolver_cfg={}, is_input=True))

        # Transform Nodes
        steps_raw = OmegaConf.to_container(cfg.steps, resolve=True)
        for step in steps_raw:  # type: ignore[union-attr]
            step_dict = dict(step)  # type: ignore[arg-type]
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
        """step の辞書から resolver_cfg 形式に正規化する。

        例:
            {"polars": {"method": "select_columns", "columns": [...]}}
            → そのまま使う

            {"output": {"columns": [...], "format": "parquet", "cv": false}}
            → {"output": {"method": "output", "columns": [...], ...}}
        """
        result: dict[str, Any] = {}
        for resolver_name, resolver_cfg in step_dict.items():
            cfg_dict = dict(resolver_cfg) if isinstance(resolver_cfg, dict) else {}
            # "output" ステップは method を自動補完
            if resolver_name == "output" and "method" not in cfg_dict:
                cfg_dict["method"] = "output"
            result[resolver_name] = cfg_dict
        return result

    def _build_cv_splits(
        self,
        cfg: DictConfig,
        input_dfs: dict[str, pl.DataFrame],
    ) -> list[tuple[list[int], list[int]]] | None:
        """cv: 設定から splits を生成する。strategy=none は None を返す。"""
        cv_cfg = cfg.get("cv", {})
        strategy = str(cv_cfg.get("strategy", "none")) if cv_cfg else "none"
        if strategy == "none":
            return None

        # 最初の input DataFrame を使って CV splits を生成
        if not input_dfs:
            return None
        first_df = next(iter(input_dfs.values()))
        n = len(first_df)
        n_splits = int(cv_cfg.get("n_splits", 5))

        if strategy == "kfold":
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            return [
                (list(map(int, train_idx)), list(map(int, test_idx)))
                for train_idx, test_idx in kf.split(range(n))
            ]
        if strategy == "time_series":
            from sklearn.model_selection import TimeSeriesSplit

            tscv = TimeSeriesSplit(n_splits=n_splits)
            return [
                (list(map(int, train_idx)), list(map(int, test_idx)))
                for train_idx, test_idx in tscv.split(range(n))
            ]

        return None

    def _build_manifest(
        self,
        job_id: str,
        commit_hash: str,
        executor_type: str,
        is_fallback: bool,
        step_results: list[StepResult],
        output_dir: Path,
    ) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "job_id": job_id,
            "timestamp": datetime.now().isoformat(),
            "commit_hash": commit_hash,
            "executor_used": "local",
            "dag_path": "pipeline_dag.html",
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
        if is_fallback:
            manifest["executor_fallback"] = True
            manifest["executor_requested"] = executor_type
        return manifest
