"""
DownloadDatasetUseCase のテスト。

なぜこのテストが必要か:
- UseCase がインフラ層（DataDownloader）に処理を委譲する責務を持つことを保証する。
- UseCase は DataDownloader の具体実装を知らず、抽象（Protocol）にのみ依存することを検証する。
- execute() が DownloadResult を返すことで、呼び出し元がログ出力や
  後続処理に利用できることを保証する。

"""

from pathlib import Path
from unittest.mock import MagicMock

from src.domain.data.downloader import DownloadResult
from src.usecase.data_acquisition.download_dataset import DownloadDatasetUseCase


class TestDownloadDatasetUseCase:
    def test_execute_calls_downloader(self) -> None:
        """execute() が downloader.download() を一度呼ぶこと。

        UseCase がインフラ層に正しく処理を委譲していることを保証する。
        UseCase 自身はダウンロードの詳細を知らなくてよい。
        """
        mock_downloader = MagicMock()
        mock_downloader.download.return_value = DownloadResult(
            output_dir=Path("/tmp/test"),
            files=[],
            commit_hash="abc123",
        )
        DownloadDatasetUseCase(mock_downloader).execute()
        mock_downloader.download.assert_called_once()

    def test_execute_returns_download_result(self) -> None:
        """execute() が DownloadResult をそのまま返すこと。

        呼び出し元（main.py）がダウンロード結果を受け取り、
        ログ出力・後続パイプラインへの連携に利用できるようにする。
        """
        expected = DownloadResult(
            output_dir=Path("/tmp/test"),
            files=[Path("/tmp/test/data.csv")],
            commit_hash="abc123",
        )
        mock_downloader = MagicMock()
        mock_downloader.download.return_value = expected
        result = DownloadDatasetUseCase(mock_downloader).execute()
        assert result == expected
