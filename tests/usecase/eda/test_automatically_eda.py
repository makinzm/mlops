"""
AutomaticallyEDAUseCase のテスト。

UseCase が複数の DataAnalyzer を順に実行し、全結果を返すことを確認する。
具体的なファイル生成はインフラ層のテストに委ねる。
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.domain.data.eda import EDAResult, FileEDAResult
from src.domain.logger.logger import AppLogger
from src.usecase.eda.automatically_eda import AutomaticallyEDAUseCase


def _make_result(commit: str = "abc123") -> EDAResult:
    file_result = FileEDAResult(
        source_path=Path("data/raw/train.csv"),
        shape=(891, 12),
        dtypes={},
        missing_counts={},
        output_files=[Path("competition/titanic_report/pandas/statistics/train_summary.parquet")],
    )
    return EDAResult(
        report_dir=Path("competition/titanic_report/20260312_1500/pandas"),
        file_results=[file_result],
        commit_hash=commit,
        readme_path=Path("competition/titanic_report/20260312_1500/pandas/README.md"),
        metainfo_path=Path("competition/titanic_report/20260312_1500/pandas/metainfo.yaml"),
    )


class TestAutomaticallyEDAUseCase:
    def test_execute_calls_all_analyzers(self) -> None:
        """execute() が全アナライザーの analyze() を 1 回ずつ呼び出すこと。"""
        a1 = MagicMock()
        a1.analyze.return_value = _make_result("abc")
        a2 = MagicMock()
        a2.analyze.return_value = _make_result("abc")
        logger = MagicMock(spec=AppLogger)

        results = AutomaticallyEDAUseCase([a1, a2], logger).execute()

        a1.analyze.assert_called_once()
        a2.analyze.assert_called_once()
        assert len(results) == 2

    def test_execute_returns_list_of_eda_results(self) -> None:
        """execute() の戻り値が list[EDAResult] であること。"""
        a1 = MagicMock()
        a1.analyze.return_value = _make_result()
        logger = MagicMock(spec=AppLogger)

        results = AutomaticallyEDAUseCase([a1], logger).execute()

        assert isinstance(results, list)
        assert results[0].commit_hash == "abc123"

    def test_execute_logs_info_on_success(self) -> None:
        """成功時に logger.info が呼ばれること。"""
        a1 = MagicMock()
        a1.analyze.return_value = _make_result()
        logger = MagicMock(spec=AppLogger)

        AutomaticallyEDAUseCase([a1], logger).execute()

        assert logger.info.call_count >= 1

    def test_execute_logs_error_and_reraises_on_failure(self) -> None:
        """analyze() が例外を送出したとき logger.error を呼んで再 raise すること。"""
        a1 = MagicMock()
        a1.analyze.side_effect = RuntimeError("disk full")
        logger = MagicMock(spec=AppLogger)

        with pytest.raises(RuntimeError):
            AutomaticallyEDAUseCase([a1], logger).execute()

        logger.error.assert_called_once()
