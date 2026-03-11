"""
データセットダウンロードのユースケース。

DataDownloader Protocol に依存し、具体的なインフラ実装を知らない。
DI により Kaggle / GCS / HuggingFace 等を差し替え可能にする。
ログの責務も UseCase に閉じ、AppLogger を DI する。
"""

from src.domain.data.downloader import DataDownloader, DownloadResult
from src.domain.logger.logger import AppLogger


class DownloadDatasetUseCase:
    def __init__(self, downloader: DataDownloader, logger: AppLogger) -> None:
        self.downloader = downloader
        self.logger = logger

    def execute(self) -> DownloadResult:
        self.logger.info("ダウンロードを開始します")
        try:
            result = self.downloader.download()
        except Exception:
            self.logger.error("ダウンロードが失敗しました", exc_info=True)
            raise
        self.logger.info(
            f"ダウンロード完了: {result.output_dir} "
            f"({len(result.files)} files, commit={result.commit_hash})"
        )
        return result
