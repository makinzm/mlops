"""
データセットダウンロードのユースケース。

DataDownloader Protocol に依存し、具体的なインフラ実装を知らない。
DI により Kaggle / GCS / HuggingFace 等を差し替え可能にする。
"""

from src.domain.data.downloader import DataDownloader, DownloadResult


class DownloadDatasetUseCase:
    def __init__(self, downloader: DataDownloader) -> None:
        self.downloader = downloader

    def execute(self) -> DownloadResult:
        return self.downloader.download()
