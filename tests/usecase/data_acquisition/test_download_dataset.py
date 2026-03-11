"""
DownloadDatasetUseCase のテスト。

なぜこのテストが必要か:
- UseCase がインフラ層（DataDownloader）に処理を委譲する責務を持つことを保証する。
- UseCase は DataDownloader の具体実装を知らず、抽象（Protocol）にのみ依存することを検証する。
- execute() が DownloadResult を返すことで、呼び出し元がログ出力や
  後続処理に利用できることを保証する。
- UseCase が開始・成功・失敗をログに記録する責務を持つことを保証する。
  ログの責務を UseCase に閉じることで、main.py がログ詳細を知る必要をなくす。

"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
        DownloadDatasetUseCase(mock_downloader, MagicMock()).execute()
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
        result = DownloadDatasetUseCase(mock_downloader, MagicMock()).execute()
        assert result == expected


class TestDownloadDatasetUseCaseLogging:
    """UseCase がログを記録する責務を持つことの検証。

    なぜこのテストが必要か:
    - ダウンロードの開始・成功・失敗をログに記録する責務は UseCase に属する。
    - main.py がログの詳細を知る必要をなくし、関心の分離を保証する。
    - logger が DI されることで、テスト時にモックへの差し替えが可能になる。
    """

    def _make_result(self) -> DownloadResult:
        return DownloadResult(
            output_dir=Path("/tmp/test"),
            files=[Path("/tmp/test/data.csv")],
            commit_hash="abc123",
        )

    def test_logger_info_called_on_start(self) -> None:
        """execute() 開始時に logger.info が呼ばれること。"""
        mock_downloader = MagicMock()
        mock_downloader.download.return_value = self._make_result()
        mock_logger = MagicMock()

        DownloadDatasetUseCase(mock_downloader, mock_logger).execute()

        assert mock_logger.info.call_count >= 1
        first_call_msg = mock_logger.info.call_args_list[0][0][0]
        assert "開始" in first_call_msg

    def test_logger_info_called_on_success(self) -> None:
        """成功時に logger.info が result 情報付きで呼ばれること。"""
        result = self._make_result()
        mock_downloader = MagicMock()
        mock_downloader.download.return_value = result
        mock_logger = MagicMock()

        DownloadDatasetUseCase(mock_downloader, mock_logger).execute()

        # 成功後のログに output_dir と commit_hash が含まれること
        all_info_calls = [str(c) for c in mock_logger.info.call_args_list]
        info_text = " ".join(all_info_calls)
        assert str(result.output_dir) in info_text
        assert result.commit_hash in info_text

    def test_logger_error_called_on_failure(self) -> None:
        """失敗時に logger.error(exc_info=True) が呼ばれること。"""
        mock_downloader = MagicMock()
        mock_downloader.download.side_effect = RuntimeError("download failed")
        mock_logger = MagicMock()

        with pytest.raises(RuntimeError):
            DownloadDatasetUseCase(mock_downloader, mock_logger).execute()

        mock_logger.error.assert_called_once()
        _, kwargs = mock_logger.error.call_args
        assert kwargs.get("exc_info") is True

    def test_exception_propagated_after_logging(self) -> None:
        """logger.error 後に例外が re-raise されること。

        エラーをログに記録しても例外を握り潰さず、呼び出し元に伝播させる必要がある。
        """
        mock_downloader = MagicMock()
        mock_downloader.download.side_effect = RuntimeError("boom")
        mock_logger = MagicMock()

        with pytest.raises(RuntimeError, match="boom"):
            DownloadDatasetUseCase(mock_downloader, mock_logger).execute()
