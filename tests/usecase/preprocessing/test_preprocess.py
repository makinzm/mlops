"""
PreprocessUseCase の E2E テスト。

なぜこのテストが必要か:
- UseCase は CLI から直接呼ばれるエントリーポイントであり、
  Hydra Config を受け取って正しくパイプラインを実行することを確認する。
- 実際のファイル I/O（parquet 書き出し）も含む E2E テストにすることで
  パイプライン全体の連携が動作していることを保証する。
- executor_fallback の記録も UseCase 経由で行われることを確認する。
"""

from pathlib import Path
from typing import Any

import polars as pl
import pytest
from omegaconf import OmegaConf

from src.usecase.preprocessing.preprocess import PreprocessUseCase


@pytest.fixture()
def train_csv(tmp_path: Path) -> Path:
    """テスト用 CSV ファイルを作成して返す。"""
    df = pl.DataFrame(
        {
            "id": list(range(10)),
            "col1": [float(i) for i in range(10)],
            "col2": [float(i * 2) for i in range(10)],
            "label": [i % 2 for i in range(10)],
        }
    )
    csv_path = tmp_path / "train.csv"
    df.write_csv(csv_path)
    return csv_path


class TestPreprocessUseCase:
    def test_execute_outputs_parquet(self, train_csv: Path, tmp_path: Path) -> None:
        """execute() が指定 output_dir に parquet を生成すること。"""
        cfg = OmegaConf.create(
            {
                "usecase": "preprocess",
                "job_id": "test_job",
                "inputs": [{"id": "raw_train", "path": str(train_csv), "format": "csv"}],
                "output_dir": str(tmp_path / "processed"),
                "executor": {"type": "local"},
                "cv": {"strategy": "none", "n_splits": 5, "time_col": None, "target_col": None},
                "steps": [
                    {
                        "id": "selected",
                        "polars": {"method": "select_columns", "columns": ["id", "col1", "label"]},
                    },
                    {
                        "id": "tabular_out",
                        "output": {
                            "columns": ["id", "col1", "label"],
                            "format": "parquet",
                            "cv": False,
                        },
                    },
                ],
                "targets": ["tabular_out"],
                "seed": 42,
            }
        )

        result = PreprocessUseCase(cfg).execute()
        assert result is not None

        # parquet が生成されていること
        job_dir = tmp_path / "processed" / "test_job"
        assert job_dir.exists()
        # タイムスタンプ付きサブディレクトリに tabular_out.parquet が存在すること
        parquet_files = list(job_dir.rglob("tabular_out.parquet"))
        assert len(parquet_files) >= 1

        # preprocess_result.yaml が生成されていること
        result_yamls = list(job_dir.rglob("preprocess_result.yaml"))
        assert len(result_yamls) >= 1

    def test_executor_fallback_recorded(self, train_csv: Path, tmp_path: Path) -> None:
        """未実装 Executor 指定時に executor_fallback=True が結果に記録されること。"""
        cfg = OmegaConf.create(
            {
                "usecase": "preprocess",
                "job_id": "fallback_job",
                "inputs": [{"id": "raw_train", "path": str(train_csv), "format": "csv"}],
                "output_dir": str(tmp_path / "processed"),
                "executor": {"type": "gcp_vertex"},  # 未実装
                "cv": {"strategy": "none", "n_splits": 5, "time_col": None, "target_col": None},
                "steps": [
                    {
                        "id": "selected",
                        "polars": {"method": "select_columns", "columns": ["id", "col1", "label"]},
                    },
                    {
                        "id": "tabular_out",
                        "output": {
                            "columns": ["id", "col1", "label"],
                            "format": "parquet",
                            "cv": False,
                        },
                    },
                ],
                "targets": ["tabular_out"],
                "seed": 42,
            }
        )

        result = PreprocessUseCase(cfg).execute()
        assert result.executor_fallback is True
        assert result.executor_requested == "gcp_vertex"
        assert result.executor_used == "local"

    def test_step_results_in_result(self, train_csv: Path, tmp_path: Path) -> None:
        """step_results に各ステップの実行状況が記録されること。"""
        cfg = OmegaConf.create(
            {
                "usecase": "preprocess",
                "job_id": "steps_job",
                "inputs": [{"id": "raw_train", "path": str(train_csv), "format": "csv"}],
                "output_dir": str(tmp_path / "processed"),
                "executor": {"type": "local"},
                "cv": {"strategy": "none", "n_splits": 5, "time_col": None, "target_col": None},
                "steps": [
                    {
                        "id": "selected",
                        "polars": {"method": "select_columns", "columns": ["id", "col1", "label"]},
                    },
                    {
                        "id": "tabular_out",
                        "output": {
                            "columns": ["id", "col1", "label"],
                            "format": "parquet",
                            "cv": False,
                        },
                    },
                ],
                "targets": ["tabular_out"],
                "seed": 42,
            }
        )
        result = PreprocessUseCase(cfg).execute()
        assert len(result.step_results) >= 1
        assert all(r.status in ("ok", "skipped", "failed") for r in result.step_results)


class TestRunsDir:
    """
    なぜこのテストが必要か:
    - `data/**` は .gitignore 対象のため、output_dir 以下のファイルは git に入らない。
    - 実験マニフェスト（preprocess_result.yaml）と DAG 可視化（pipeline_dag.html）は
      git で履歴管理したいため、git-tracked な `runs_dir/` に保存する必要がある。
    - runs_dir を分離することで「データ（大容量）」と「メタデータ（小容量）」を
      明確に分離でき、レビュアーが実験記録を git 上で追跡できるようになる。
    """

    @pytest.fixture()
    def base_cfg(self, train_csv: Path, tmp_path: Path) -> dict[str, Any]:
        return {
            "usecase": "preprocess",
            "job_id": "runs_test_job",
            "inputs": [{"id": "raw_train", "path": str(train_csv), "format": "csv"}],
            "output_dir": str(tmp_path / "processed"),
            "runs_dir": str(tmp_path / "runs"),
            "executor": {"type": "local"},
            "cv": {"strategy": "none", "n_splits": 5, "time_col": None, "target_col": None},
            "steps": [
                {
                    "id": "selected",
                    "polars": {"method": "select_columns", "columns": ["id", "col1", "label"]},
                },
                {
                    "id": "tabular_out",
                    "output": {
                        "columns": ["id", "col1", "label"],
                        "format": "parquet",
                        "cv": False,
                    },
                },
            ],
            "targets": ["tabular_out"],
            "seed": 42,
        }

    def test_preprocess_result_yaml_saved_in_runs_dir(
        self, base_cfg: dict[str, Any], tmp_path: Path
    ) -> None:
        """runs_dir が指定された場合、preprocess_result.yaml は runs_dir/ 以下に保存される。"""
        cfg = OmegaConf.create(base_cfg)
        PreprocessUseCase(cfg).execute()

        runs_dir = tmp_path / "runs"
        result_yamls = list(runs_dir.rglob("preprocess_result.yaml"))
        assert len(result_yamls) >= 1, "preprocess_result.yaml が runs_dir/ 以下に存在すること"

        # output_dir 側には yaml が存在しないこと
        output_dir = tmp_path / "processed"
        output_yamls = list(output_dir.rglob("preprocess_result.yaml"))
        assert len(output_yamls) == 0, "preprocess_result.yaml が output_dir/ 以下に存在しないこと"

    def test_pipeline_dag_html_saved_in_runs_dir(
        self, base_cfg: dict[str, Any], tmp_path: Path
    ) -> None:
        """runs_dir が指定された場合、pipeline_dag.html は runs_dir/ 以下に保存される。"""
        cfg = OmegaConf.create(base_cfg)
        PreprocessUseCase(cfg).execute()

        runs_dir = tmp_path / "runs"
        html_files = list(runs_dir.rglob("pipeline_dag.html"))
        assert len(html_files) >= 1, "pipeline_dag.html が runs_dir/ 以下に存在すること"

        # output_dir 側には html が存在しないこと
        output_dir = tmp_path / "processed"
        output_htmls = list(output_dir.rglob("pipeline_dag.html"))
        assert len(output_htmls) == 0, "pipeline_dag.html が output_dir/ 以下に存在しないこと"

    def test_preprocess_result_yaml_contains_output_dir(
        self, base_cfg: dict[str, Any], tmp_path: Path
    ) -> None:
        """preprocess_result.yaml に output_dir フィールドが含まれる（データの場所を記録）。"""
        import yaml

        cfg = OmegaConf.create(base_cfg)
        PreprocessUseCase(cfg).execute()

        runs_dir = tmp_path / "runs"
        result_yamls = list(runs_dir.rglob("preprocess_result.yaml"))
        assert len(result_yamls) >= 1

        with open(result_yamls[0]) as f:
            manifest = yaml.safe_load(f)

        assert "output_dir" in manifest, "preprocess_result.yaml に output_dir フィールドがあること"
        assert manifest["output_dir"], "output_dir フィールドが空でないこと"
