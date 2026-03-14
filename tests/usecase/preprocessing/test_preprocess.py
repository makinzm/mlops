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
