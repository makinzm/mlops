"""
KaggleDownloader のテスト。

なぜこのテストが必要か:
- KaggleDownloader は外部 API に依存するため、CI でも動作するようモックで検証する。
- dataset / competition の両モードで正しい API メソッドが呼ばれることを保証する。
- 不正な mode に対して明示的にエラーを出すことでサイレントな不具合を防ぐ。
- DownloadResult に commit_hash が含まれることで DoD（CommitHash 記録）を満たすことを保証する。
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig, OmegaConf
from src.infrastructure.downloader.kaggle import KaggleDownloader


@pytest.fixture
def dataset_cfg(tmp_path: Path) -> DictConfig:
    return OmegaConf.create(
        {
            "data_from": "kaggle",
            "output_dir": str(tmp_path),
            "unzip": True,
            "force": False,
            "kaggle": {
                "mode": "dataset",
                "dataset": "testuser/test-dataset",
                "competition": None,
            },
        }
    )


@pytest.fixture
def competition_cfg(tmp_path: Path) -> DictConfig:
    return OmegaConf.create(
        {
            "data_from": "kaggle",
            "output_dir": str(tmp_path),
            "unzip": True,
            "force": False,
            "kaggle": {
                "mode": "competition",
                "dataset": None,
                "competition": "test-competition",
            },
        }
    )


class TestKaggleDownloaderInit:
    def test_authenticate_called_on_init(self, dataset_cfg: DictConfig) -> None:
        """KaggleDownloader 初期化時に Kaggle API の認証が行われること。

        認証を初期化タイミングで実行することで、実行前に認証エラーを早期発見できる。
        外部 API 依存のためモックで代替する。
        """
        with patch("src.infrastructure.downloader.kaggle.KaggleApiExtended") as mock_cls:
            mock_api = MagicMock()
            mock_cls.return_value = mock_api
            KaggleDownloader(dataset_cfg)
            mock_api.authenticate.assert_called_once()


class TestKaggleDownloaderDataset:
    def test_download_dataset_calls_api(self, dataset_cfg: DictConfig) -> None:
        """dataset モードで dataset_download_files が正しい引数で呼ばれること。

        API への引数マッピングが正しいことを保証する。
        特に dataset 名と output_dir の対応を検証する。
        """
        with patch("src.infrastructure.downloader.kaggle.KaggleApiExtended") as mock_cls:
            mock_api = MagicMock()
            mock_cls.return_value = mock_api
            KaggleDownloader(dataset_cfg).download()
            mock_api.dataset_download_files.assert_called_once()
            call_args = mock_api.dataset_download_files.call_args
            assert "testuser/test-dataset" in str(call_args)

    def test_download_result_has_output_dir(self, dataset_cfg: DictConfig) -> None:
        """DownloadResult の output_dir が設定値と一致すること。

        呼び出し元がダウンロード先を把握できるようにするための検証。
        """
        with patch("src.infrastructure.downloader.kaggle.KaggleApiExtended") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = KaggleDownloader(dataset_cfg).download()
            assert result.output_dir == Path(dataset_cfg.output_dir)

    def test_download_result_contains_commit_hash(self, dataset_cfg: DictConfig) -> None:
        """DownloadResult に commit_hash が含まれること。

        DoD 要件: 全てのコードは Git の CommitHash が記録されること。
        どのコードでダウンロードしたかを追跡可能にし、再現性を担保する。
        """
        with patch("src.infrastructure.downloader.kaggle.KaggleApiExtended") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = KaggleDownloader(dataset_cfg).download()
            assert isinstance(result.commit_hash, str)
            assert len(result.commit_hash) > 0


class TestKaggleDownloaderCompetition:
    def test_download_competition_calls_api(self, competition_cfg: DictConfig) -> None:
        """competition モードで competition_download_files が呼ばれること。

        dataset / competition の両モードで正しい API メソッドが選択されることを保証する。
        """
        with patch("src.infrastructure.downloader.kaggle.KaggleApiExtended") as mock_cls:
            mock_api = MagicMock()
            mock_cls.return_value = mock_api
            KaggleDownloader(competition_cfg).download()
            mock_api.competition_download_files.assert_called_once()

    def test_download_competition_result_has_output_dir(self, competition_cfg: DictConfig) -> None:
        """competition モードでも DownloadResult の output_dir が設定値と一致すること。"""
        with patch("src.infrastructure.downloader.kaggle.KaggleApiExtended") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = KaggleDownloader(competition_cfg).download()
            assert result.output_dir == Path(competition_cfg.output_dir)


class TestKaggleDownloaderInvalidMode:
    def test_invalid_mode_raises_value_error(self, dataset_cfg: DictConfig) -> None:
        """不明な mode に対して ValueError が発生すること。

        予期しない設定値に対して明示的にエラーを発生させることで、
        サイレントな不具合（無音でスキップ等）を防ぐ。
        """
        cfg = OmegaConf.merge(dataset_cfg, {"kaggle": {"mode": "unknown_mode"}})
        with patch("src.infrastructure.downloader.kaggle.KaggleApiExtended") as mock_cls:
            mock_cls.return_value = MagicMock()
            downloader = KaggleDownloader(cfg)
            with pytest.raises(ValueError, match="Unknown mode"):
                downloader.download()
