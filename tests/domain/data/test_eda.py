"""
domain/data/eda.py のドメイン定義テスト。

EDAResult / FileEDAResult / AnalysisStep / DataAnalyzer Protocol が
正しく定義されていることを確認する。
型チェックのみで副作用のないテストなので高速に実行できる。
"""

from pathlib import Path

import pytest

from src.domain.data.eda import AnalysisStep, DataAnalyzer, EDAResult, FileEDAResult


class TestAnalysisStep:
    def test_type_field_required(self) -> None:
        step = AnalysisStep(type="basic_stats")
        assert step.type == "basic_stats"

    def test_params_defaults_to_empty_dict(self) -> None:
        step = AnalysisStep(type="group_stats")
        assert step.params == {}

    def test_params_can_be_set(self) -> None:
        step = AnalysisStep(type="group_stats", params={"group_by": "Survived"})
        assert step.params["group_by"] == "Survived"


class TestFileEDAResult:
    def test_fields_accessible(self) -> None:
        result = FileEDAResult(
            source_path=Path("data/raw/train.csv"),
            shape=(891, 12),
            dtypes={"PassengerId": "int64", "Name": "object"},
            missing_counts={"Age": 177, "Cabin": 687},
            output_files=[Path("competition/titanic_report/statistics/train_summary.parquet")],
        )
        assert result.shape == (891, 12)
        assert result.missing_counts["Age"] == 177
        assert len(result.output_files) == 1


class TestEDAResult:
    def test_fields_accessible(self) -> None:
        file_result = FileEDAResult(
            source_path=Path("data/raw/train.csv"),
            shape=(891, 12),
            dtypes={},
            missing_counts={},
            output_files=[],
        )
        result = EDAResult(
            report_dir=Path("competition/titanic_report/20260312_1500"),
            file_results=[file_result],
            commit_hash="abc123",
            readme_path=Path("competition/titanic_report/20260312_1500/README.md"),
            metainfo_path=Path("competition/titanic_report/20260312_1500/metainfo.yaml"),
        )
        assert result.commit_hash == "abc123"
        assert len(result.file_results) == 1


class TestDataAnalyzerProtocol:
    def test_class_satisfying_protocol_is_accepted(self) -> None:
        """DataAnalyzer Protocol を満たすクラスが isinstance チェックを通ること。"""

        class FakeAnalyzer:
            def analyze(self) -> EDAResult:
                return EDAResult(
                    report_dir=Path("."),
                    file_results=[],
                    commit_hash="abc",
                    readme_path=Path("README.md"),
                    metainfo_path=Path("metainfo.yaml"),
                )

        assert isinstance(FakeAnalyzer(), DataAnalyzer)

    def test_class_missing_analyze_fails_protocol(self) -> None:
        """analyze() を持たないクラスは DataAnalyzer を満たさない。"""

        class NotAnAnalyzer:
            pass

        assert not isinstance(NotAnAnalyzer(), DataAnalyzer)
