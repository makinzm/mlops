"""
AutomaticallyEDAUseCase のテスト。

UseCase が DataAnalyzer Protocol に依存し、analyze() を呼び出して
結果をログに記録することを確認する。
具体的なファイル生成はインフラ層のテストに委ねる。
"""

from pathlib import Path
from unittest.mock import MagicMock

from src.domain.data.eda import EDAResult, FileEDAResult
from src.domain.logger.logger import AppLogger
from src.usecase.eda.automatically_eda import AutomaticallyEDAUseCase


def _make_result() -> EDAResult:
    file_result = FileEDAResult(
        source_path=Path("data/raw/train.csv"),
        shape=(891, 12),
        dtypes={},
        missing_counts={},
        output_files=[Path("competition/titanic_report/statistics/train_summary.parquet")],
    )
    return EDAResult(
        report_dir=Path("competition/titanic_report/20260312_1500"),
        file_results=[file_result],
        commit_hash="abc123",
        readme_path=Path("competition/titanic_report/20260312_1500/README.md"),
        metainfo_path=Path("competition/titanic_report/20260312_1500/metainfo.yaml"),
    )


class TestAutomaticallyEDAUseCase:
    def test_execute_calls_analyzer(self) -> None:
        """execute() が DataAnalyzer.analyze() を 1 回呼び出すこと。"""
        analyzer = MagicMock()
        analyzer.analyze.return_value = _make_result()
        logger = MagicMock(spec=AppLogger)

        usecase = AutomaticallyEDAUseCase(analyzer, logger)
        result = usecase.execute()

        analyzer.analyze.assert_called_once()
        assert result.commit_hash == "abc123"

    def test_execute_logs_info_on_success(self) -> None:
        """成功時に logger.info が呼ばれること。"""
        analyzer = MagicMock()
        analyzer.analyze.return_value = _make_result()
        logger = MagicMock(spec=AppLogger)

        AutomaticallyEDAUseCase(analyzer, logger).execute()

        assert logger.info.call_count >= 1

    def test_execute_logs_error_and_reraises_on_failure(self) -> None:
        """analyze() が例外を送出したとき logger.error を呼んで再 raise すること。"""
        analyzer = MagicMock()
        analyzer.analyze.side_effect = RuntimeError("disk full")
        logger = MagicMock(spec=AppLogger)

        import pytest

        with pytest.raises(RuntimeError):
            AutomaticallyEDAUseCase(analyzer, logger).execute()

        logger.error.assert_called_once()
