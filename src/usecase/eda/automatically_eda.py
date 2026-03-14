"""
自動 EDA ユースケース。

複数の DataAnalyzer を順に実行し、各結果を返す。
インフラ層の選択（pandas / polars 等）は main.py の DI 層で行う。
"""

from src.domain.data.eda import DataAnalyzer, EDAResult
from src.domain.logger.logger import AppLogger


class AutomaticallyEDAUseCase:
    def __init__(self, analyzers: list[DataAnalyzer], logger: AppLogger) -> None:
        self.analyzers = analyzers
        self.logger = logger

    def execute(self) -> list[EDAResult]:
        self.logger.info(f"EDA を開始します（アナライザー数: {len(self.analyzers)}）")
        results: list[EDAResult] = []
        for analyzer in self.analyzers:
            try:
                result = analyzer.analyze()
            except Exception:
                self.logger.error("EDA が失敗しました", exc_info=True)
                raise
            self.logger.info(
                f"EDA 完了: {result.report_dir} "
                f"({len(result.file_results)} files, commit={result.commit_hash})"
            )
            results.append(result)
        return results
