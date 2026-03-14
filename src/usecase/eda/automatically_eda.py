"""
自動 EDA ユースケース。

DataAnalyzer Protocol に依存し、具体的なインフラ実装を知らない。
"""

from src.domain.data.eda import DataAnalyzer, EDAResult
from src.domain.logger.logger import AppLogger


class AutomaticallyEDAUseCase:
    def __init__(self, analyzer: DataAnalyzer, logger: AppLogger) -> None:
        self.analyzer = analyzer
        self.logger = logger

    def execute(self) -> EDAResult:
        self.logger.info("EDA を開始します")
        try:
            result = self.analyzer.analyze()
        except Exception:
            self.logger.error("EDA が失敗しました", exc_info=True)
            raise
        self.logger.info(
            f"EDA 完了: {result.report_dir} "
            f"({len(result.file_results)} files, commit={result.commit_hash})"
        )
        return result
