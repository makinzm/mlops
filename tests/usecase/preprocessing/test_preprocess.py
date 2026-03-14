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


class TestMetadataInOutputDir:
    """
    なぜこのテストが必要か:
    - preprocess_result.yaml と pipeline_dag.html はメタデータとして git 管理したい。
    - .gitignore の data/** に対して *.yaml / *.html を除外対象外にすることで
      output_dir（data/processed/）以下に保存しつつ git に入れる設計。
    - runs/ のような別ディレクトリは不要で、データとメタデータを同じ場所に置く。
    """

    @pytest.fixture()
    def base_cfg(self, train_csv: Path, tmp_path: Path) -> dict[str, Any]:
        return {
            "usecase": "preprocess",
            "job_id": "metadata_test_job",
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

    def test_preprocess_result_yaml_saved_in_output_dir(
        self, base_cfg: dict[str, Any], tmp_path: Path
    ) -> None:
        """preprocess_result.yaml は output_dir/ 以下に保存される。"""
        cfg = OmegaConf.create(base_cfg)
        PreprocessUseCase(cfg).execute()

        output_dir = tmp_path / "processed"
        result_yamls = list(output_dir.rglob("preprocess_result.yaml"))
        assert len(result_yamls) >= 1, "preprocess_result.yaml が output_dir/ 以下に存在すること"

    def test_pipeline_dag_html_saved_in_output_dir(
        self, base_cfg: dict[str, Any], tmp_path: Path
    ) -> None:
        """pipeline_dag.html は output_dir/ 以下に保存される。"""
        cfg = OmegaConf.create(base_cfg)
        PreprocessUseCase(cfg).execute()

        output_dir = tmp_path / "processed"
        html_files = list(output_dir.rglob("pipeline_dag.html"))
        assert len(html_files) >= 1, "pipeline_dag.html が output_dir/ 以下に存在すること"

    def test_preprocess_result_yaml_contains_output_dir(
        self, base_cfg: dict[str, Any], tmp_path: Path
    ) -> None:
        """preprocess_result.yaml に output_dir フィールドが含まれる（データの場所を記録）。"""
        import yaml

        cfg = OmegaConf.create(base_cfg)
        PreprocessUseCase(cfg).execute()

        output_dir = tmp_path / "processed"
        result_yamls = list(output_dir.rglob("preprocess_result.yaml"))
        assert len(result_yamls) >= 1

        with open(result_yamls[0]) as f:
            manifest = yaml.safe_load(f)

        assert "output_dir" in manifest, "preprocess_result.yaml に output_dir フィールドがあること"
        assert manifest["output_dir"], "output_dir フィールドが空でないこと"


class TestCvStrategy:
    """
    なぜこのテストが必要か:
    - kfold/time_series しか実装されておらず、ターゲットの分布を考慮した分割や
      グループを考慮した分割が必要なケースに対応できない。
    - Titanic のような分類タスクでは stratified_kfold が必須であり、
      分割がクラス比率を保持していることを確認する必要がある。
    - group_kfold/leave_one_group_out は時系列や患者 ID など
      データリーク防止のために不可欠な戦略。
    - input_id 指定により、複数入力がある場合に CV 分割する対象を明示できる。
    """

    @pytest.fixture()
    def df_with_label_and_group(self, tmp_path: Path) -> Path:
        """label（0/1 各5行）と group（0-4 各2行）を持つ CSV を作成する。"""
        df = pl.DataFrame(
            {
                "id": list(range(10)),
                "feature": [float(i) for i in range(10)],
                "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                "group": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            }
        )
        csv_path = tmp_path / "data.csv"
        df.write_csv(csv_path)
        return csv_path

    def _make_cfg(
        self,
        csv_path: Path,
        tmp_path: Path,
        strategy: str,
        n_splits: int = 5,
        target_col: str | None = None,
        group_col: str | None = None,
        input_id: str | None = None,
    ) -> Any:
        return OmegaConf.create(
            {
                "usecase": "preprocess",
                "job_id": f"cv_{strategy}_test",
                "inputs": [{"id": "raw", "path": str(csv_path), "format": "csv"}],
                "output_dir": str(tmp_path / "processed"),
                "executor": {"type": "local"},
                "cv": {
                    "strategy": strategy,
                    "n_splits": n_splits,
                    "time_col": None,
                    "target_col": target_col,
                    "group_col": group_col,
                    "input_id": input_id,
                },
                "steps": [
                    {
                        "id": "out",
                        "output": {
                            "columns": ["id", "feature"],
                            "format": "parquet",
                            "cv": True,
                        },
                    },
                ],
                "targets": ["out"],
                "seed": 42,
            }
        )

    def test_cv_stratified_kfold(self, df_with_label_and_group: Path, tmp_path: Path) -> None:
        """stratified_kfold: target_col を指定するとクラス比率を保った 5 splits が生成される。"""
        cfg = self._make_cfg(
            df_with_label_and_group, tmp_path, "stratified_kfold", target_col="label"
        )
        result = PreprocessUseCase(cfg).execute()
        assert result.n_splits == 5, "stratified_kfold で n_splits=5 の splits が生成されること"

    def test_cv_group_kfold(self, df_with_label_and_group: Path, tmp_path: Path) -> None:
        """group_kfold: group_col を指定するとグループ境界を守った splits が生成される。"""
        cfg = self._make_cfg(df_with_label_and_group, tmp_path, "group_kfold", group_col="group")
        result = PreprocessUseCase(cfg).execute()
        assert result.n_splits is not None
        assert result.n_splits >= 1, "group_kfold で splits が生成されること"

    def test_cv_stratified_group_kfold(self, df_with_label_and_group: Path, tmp_path: Path) -> None:
        """stratified_group_kfold: target_col と group_col を両方指定すると splits が生成される。"""
        cfg = self._make_cfg(
            df_with_label_and_group,
            tmp_path,
            "stratified_group_kfold",
            n_splits=3,
            target_col="label",
            group_col="group",
        )
        result = PreprocessUseCase(cfg).execute()
        assert result.n_splits is not None
        assert result.n_splits >= 1, "stratified_group_kfold で splits が生成されること"

    def test_cv_leave_one_group_out(self, df_with_label_and_group: Path, tmp_path: Path) -> None:
        """leave_one_group_out: group 数（5）と同数の splits が生成される。"""
        cfg = self._make_cfg(
            df_with_label_and_group, tmp_path, "leave_one_group_out", group_col="group"
        )
        result = PreprocessUseCase(cfg).execute()
        # グループ数 = 5 なので splits も 5
        assert result.n_splits == 5, "leave_one_group_out でグループ数の splits が生成されること"

    def test_cv_input_id_selects_correct_df(self, tmp_path: Path) -> None:
        """input_id が指定された場合、その input の DataFrame を使って splits が生成される。"""
        # 10行の df_a と 20行の df_b を用意する
        df_a = pl.DataFrame(
            {
                "id": list(range(10)),
                "feature": [float(i) for i in range(10)],
                "label": [i % 2 for i in range(10)],
            }
        )
        df_b = pl.DataFrame(
            {
                "id": list(range(20)),
                "feature": [float(i) for i in range(20)],
                "label": [i % 2 for i in range(20)],
            }
        )
        csv_a = tmp_path / "a.csv"
        csv_b = tmp_path / "b.csv"
        df_a.write_csv(csv_a)
        df_b.write_csv(csv_b)

        cfg = OmegaConf.create(
            {
                "usecase": "preprocess",
                "job_id": "input_id_test",
                "inputs": [
                    {"id": "df_a", "path": str(csv_a), "format": "csv"},
                    {"id": "df_b", "path": str(csv_b), "format": "csv"},
                ],
                "output_dir": str(tmp_path / "processed"),
                "executor": {"type": "local"},
                "cv": {
                    "strategy": "stratified_kfold",
                    "n_splits": 5,
                    "time_col": None,
                    "target_col": "label",
                    "group_col": None,
                    "input_id": "df_a",  # 10 行の df_a を使う
                },
                "steps": [
                    {
                        "id": "out_a",
                        "from": "df_a",
                        "output": {
                            "columns": ["id", "feature"],
                            "format": "parquet",
                            "cv": True,
                        },
                    },
                ],
                "targets": ["out_a"],
                "seed": 42,
            }
        )
        result = PreprocessUseCase(cfg).execute()
        # df_a (10行) で stratified_kfold n_splits=5 → 5 splits
        assert result.n_splits == 5, "input_id で指定した df_a の行数で splits が生成されること"
