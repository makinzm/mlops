"""
KaggleDownloader のテスト。

なぜこのテストが必要か:
- KaggleDownloader は外部 API に依存するため、CI でも動作するようモックで検証する。
- dataset / competition の両モードで正しい API メソッドが呼ばれることを保証する。
- 不正な mode に対して明示的にエラーを出すことでサイレントな不具合を防ぐ。
- DownloadResult に commit_hash が含まれることで DoD（CommitHash 記録）を満たすことを保証する。
- GitRepository が DI されることで、git 操作がインフラ層に閉じていることを保証する。
- metadata.yaml が保存されることで、将来のデータ参照時に「いつ・どの設定・どのコードで
  取得したか」を追跡できることを保証する。
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from omegaconf import DictConfig, OmegaConf

from src.infrastructure.downloader.kaggle import KaggleDownloader


@pytest.fixture
def mock_git_repo() -> MagicMock:
    git_repo = MagicMock()
    git_repo.get_commit_hash.return_value = "abc123"
    return git_repo


@pytest.fixture
def dataset_cfg(tmp_path: Path) -> DictConfig:
    return OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "unzip": True,
            "force": False,
            "downloader": {
                "type": "kaggle",
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
            "output_dir": str(tmp_path),
            "unzip": True,
            "force": False,
            "downloader": {
                "type": "kaggle",
                "mode": "competition",
                "dataset": None,
                "competition": "test-competition",
            },
        }
    )


@pytest.fixture
def dataset_cfg_no_unzip(tmp_path: Path) -> DictConfig:
    return OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "unzip": False,
            "force": False,
            "downloader": {
                "type": "kaggle",
                "mode": "dataset",
                "dataset": "testuser/test-dataset",
                "competition": None,
            },
        }
    )


@pytest.fixture
def competition_cfg_no_unzip(tmp_path: Path) -> DictConfig:
    return OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "unzip": False,
            "force": False,
            "downloader": {
                "type": "kaggle",
                "mode": "competition",
                "dataset": None,
                "competition": "test-competition",
            },
        }
    )


class TestKaggleDownloaderInit:
    def test_authenticate_called_on_init(
        self, dataset_cfg: DictConfig, mock_git_repo: MagicMock
    ) -> None:
        """KaggleDownloader 初期化時に Kaggle API の認証が行われること。

        認証を初期化タイミングで実行することで、実行前に認証エラーを早期発見できる。
        外部 API 依存のためモックで代替する。
        """
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_api = MagicMock()
            mock_cls.return_value = mock_api
            KaggleDownloader(dataset_cfg, mock_git_repo)
            mock_api.authenticate.assert_called_once()


class TestKaggleDownloaderDataset:
    def test_download_dataset_calls_api(
        self, dataset_cfg: DictConfig, mock_git_repo: MagicMock
    ) -> None:
        """dataset モードで dataset_download_files が正しい引数で呼ばれること。

        API への引数マッピングが正しいことを保証する。
        特に dataset 名と output_dir の対応を検証する。
        """
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_api = MagicMock()
            mock_cls.return_value = mock_api
            KaggleDownloader(dataset_cfg, mock_git_repo).download()
            mock_api.dataset_download_files.assert_called_once()
            call_args = mock_api.dataset_download_files.call_args
            assert "testuser/test-dataset" in str(call_args)

    def test_download_result_has_output_dir(
        self, dataset_cfg: DictConfig, mock_git_repo: MagicMock
    ) -> None:
        """DownloadResult の output_dir が設定値と一致すること。"""
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = KaggleDownloader(dataset_cfg, mock_git_repo).download()
            assert result.output_dir == Path(dataset_cfg.output_dir)

    def test_download_result_contains_commit_hash(
        self, dataset_cfg: DictConfig, mock_git_repo: MagicMock
    ) -> None:
        """DownloadResult に commit_hash が含まれること。

        DoD 要件: 全てのコードは Git の CommitHash が記録されること。
        GitRepository 経由で取得することでインフラ依存が downloader に漏れないことも確認する。
        """
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = KaggleDownloader(dataset_cfg, mock_git_repo).download()
            assert result.commit_hash == "abc123"
            mock_git_repo.get_commit_hash.assert_called()

    def test_setup_data_dir_called_on_download(
        self, dataset_cfg: DictConfig, mock_git_repo: MagicMock
    ) -> None:
        """ダウンロード時に git_repo.setup_data_dir が呼ばれること。

        .gitkeep / .gitignore の生成責務が GitRepository に委譲されていることを保証する。
        """
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_cls.return_value = MagicMock()
            KaggleDownloader(dataset_cfg, mock_git_repo).download()
            mock_git_repo.setup_data_dir.assert_called_once_with(Path(dataset_cfg.output_dir))


class TestKaggleDownloaderMetadata:
    """metadata.yaml の保存を検証するテスト。

    なぜこのテストが必要か:
    - ダウンロード後に「いつ・どのコード・どの設定で取得したか」を追跡するためのメタデータが
      output_dir に保存されることを保証する。
    - metadata.yaml が git 追跡対象となることで、データ本体を削除しても設定履歴が残る。
    - metadata.yaml が DownloadResult.files に含まれないことで、
      後続パイプラインがデータファイルのみを処理できることを保証する。
    """

    def test_metadata_file_created_after_download(
        self, dataset_cfg: DictConfig, mock_git_repo: MagicMock
    ) -> None:
        """ダウンロード後に metadata.yaml が output_dir に作成されること。"""
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = KaggleDownloader(dataset_cfg, mock_git_repo).download()
            assert (result.output_dir / "metadata.yaml").exists()

    def test_metadata_contains_commit_hash(
        self, dataset_cfg: DictConfig, mock_git_repo: MagicMock
    ) -> None:
        """metadata.yaml に commit_hash が含まれること。"""
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = KaggleDownloader(dataset_cfg, mock_git_repo).download()
            metadata = yaml.safe_load((result.output_dir / "metadata.yaml").read_text())
            assert metadata["commit_hash"] == "abc123"

    def test_metadata_contains_config(
        self, dataset_cfg: DictConfig, mock_git_repo: MagicMock
    ) -> None:
        """metadata.yaml に Hydra config が含まれること。"""
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = KaggleDownloader(dataset_cfg, mock_git_repo).download()
            metadata = yaml.safe_load((result.output_dir / "metadata.yaml").read_text())
            assert "config" in metadata
            assert metadata["config"]["downloader"]["dataset"] == "testuser/test-dataset"

    def test_metadata_contains_downloaded_at(
        self, dataset_cfg: DictConfig, mock_git_repo: MagicMock
    ) -> None:
        """metadata.yaml にダウンロード日時が含まれること。"""
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = KaggleDownloader(dataset_cfg, mock_git_repo).download()
            metadata = yaml.safe_load((result.output_dir / "metadata.yaml").read_text())
            assert "downloaded_at" in metadata

    def test_metadata_not_in_result_files(
        self, dataset_cfg: DictConfig, mock_git_repo: MagicMock
    ) -> None:
        """metadata.yaml が DownloadResult.files に含まれないこと。

        後続パイプラインがデータファイルのみを受け取れるよう、
        生成ファイル（metadata.yaml）はファイルリストから除外する。
        """
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = KaggleDownloader(dataset_cfg, mock_git_repo).download()
            assert all(f.name != "metadata.yaml" for f in result.files)


class TestKaggleDownloaderCompetition:
    def test_download_competition_calls_api(
        self, competition_cfg: DictConfig, mock_git_repo: MagicMock
    ) -> None:
        """competition モードで competition_download_files が呼ばれること。

        dataset / competition の両モードで正しい API メソッドが選択されることを保証する。
        """
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_api = MagicMock()
            mock_cls.return_value = mock_api
            KaggleDownloader(competition_cfg, mock_git_repo).download()
            mock_api.competition_download_files.assert_called_once()

    def test_download_competition_result_has_output_dir(
        self, competition_cfg: DictConfig, mock_git_repo: MagicMock
    ) -> None:
        """competition モードでも DownloadResult の output_dir が設定値と一致すること。"""
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = KaggleDownloader(competition_cfg, mock_git_repo).download()
            assert result.output_dir == Path(competition_cfg.output_dir)

    def test_competition_metadata_created(
        self, competition_cfg: DictConfig, mock_git_repo: MagicMock
    ) -> None:
        """competition モードでも metadata.yaml が作成されること。"""
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = KaggleDownloader(competition_cfg, mock_git_repo).download()
            assert (result.output_dir / "metadata.yaml").exists()


class TestKaggleDownloaderUnzip:
    """unzip オプションの動作を検証するテスト。

    なぜこのテストが必要か:
    - competition_download_files は unzip パラメータを持たないため、
      API に依存せず手動で ZIP 展開する必要がある。
    - dataset_download_files の unzip=True は「ダウンロードが発生した場合のみ」展開するため、
      既存 ZIP があるケースで展開が起きない。これを手動展開で解消する。
    - テストでは Kaggle API のダウンロード副作用として ZIP ファイルを生成し、
      展開後の CSV が存在すること・ZIP が削除されることを確認する。
    """

    def _make_zip_side_effect(self, tmp_path: Path) -> object:
        """Kaggle API 呼び出し時に ZIP ファイルを生成するサイドエフェクト関数を返す。"""
        import zipfile

        def _side_effect(*args: object, **kwargs: object) -> None:
            zip_path = tmp_path / "data.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("train.csv", "col1,col2\n1,2\n")

        return _side_effect

    def test_competition_unzips_when_unzip_true(
        self, competition_cfg: DictConfig, mock_git_repo: MagicMock, tmp_path: Path
    ) -> None:
        """unzip=True のとき competition モードで ZIP が展開されること。

        competition_download_files は unzip パラメータを持たないため、
        ダウンロード後に手動で展開しなければ後続の前処理パイプラインが ZIP を直接処理することになる。
        """
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_api = MagicMock()
            mock_api.competition_download_files.side_effect = self._make_zip_side_effect(tmp_path)
            mock_cls.return_value = mock_api
            KaggleDownloader(competition_cfg, mock_git_repo).download()
        assert (tmp_path / "train.csv").exists()
        assert not (tmp_path / "data.zip").exists()

    def test_competition_does_not_unzip_when_unzip_false(
        self,
        competition_cfg_no_unzip: DictConfig,
        mock_git_repo: MagicMock,
        tmp_path: Path,
    ) -> None:
        """unzip=False のとき competition モードで ZIP が展開されないこと。

        展開するかどうかはユーザーが config で制御できる必要がある。
        unzip=False のときは ZIP をそのまま残す。
        """
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_api = MagicMock()
            mock_api.competition_download_files.side_effect = self._make_zip_side_effect(tmp_path)
            mock_cls.return_value = mock_api
            KaggleDownloader(competition_cfg_no_unzip, mock_git_repo).download()
        assert (tmp_path / "data.zip").exists()
        assert not (tmp_path / "train.csv").exists()

    def test_dataset_unzips_when_unzip_true(
        self, dataset_cfg: DictConfig, mock_git_repo: MagicMock, tmp_path: Path
    ) -> None:
        """unzip=True のとき dataset モードで ZIP が展開されること。

        dataset_download_files の unzip=True は既存ファイルがある場合に展開をスキップする。
        手動展開により、再実行時も確実に ZIP が展開されることを保証する。
        """
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_api = MagicMock()
            mock_api.dataset_download_files.side_effect = self._make_zip_side_effect(tmp_path)
            mock_cls.return_value = mock_api
            KaggleDownloader(dataset_cfg, mock_git_repo).download()
        assert (tmp_path / "train.csv").exists()
        assert not (tmp_path / "data.zip").exists()

    def test_dataset_does_not_unzip_when_unzip_false(
        self,
        dataset_cfg_no_unzip: DictConfig,
        mock_git_repo: MagicMock,
        tmp_path: Path,
    ) -> None:
        """unzip=False のとき dataset モードで ZIP が展開されないこと。"""
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_api = MagicMock()
            mock_api.dataset_download_files.side_effect = self._make_zip_side_effect(tmp_path)
            mock_cls.return_value = mock_api
            KaggleDownloader(dataset_cfg_no_unzip, mock_git_repo).download()
        assert (tmp_path / "data.zip").exists()
        assert not (tmp_path / "train.csv").exists()


class TestKaggleDownloaderForceCheck:
    """force=False のとき既存ファイルがある場合の挙動を検証するテスト。

    なぜこのテストが必要か:
    - Kaggle API は force=False のとき既存ファイルがあると静かにスキップする。
      ユーザーは「ダウンロードされた」と思い込んで古いデータを使い続けるリスクがある。
    - 明示的に FileExistsError を送出することで、ユーザーが意図的に force=true を
      指定しない限り既存データを上書きしないことを保証する。
    - .gitkeep 等の管理ファイルはデータファイルではないため除外する。
    """

    def test_download_dataset_fails_when_files_exist_and_force_false(
        self, dataset_cfg: DictConfig, mock_git_repo: MagicMock, tmp_path: Path
    ) -> None:
        """dataset モードで force=False のとき既存データファイルがあれば FileExistsError になること。"""
        (tmp_path / "train.csv").write_text("existing data")
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_cls.return_value = MagicMock()
            with pytest.raises(FileExistsError, match="force=true"):
                KaggleDownloader(dataset_cfg, mock_git_repo).download()

    def test_download_competition_fails_when_files_exist_and_force_false(
        self, competition_cfg: DictConfig, mock_git_repo: MagicMock, tmp_path: Path
    ) -> None:
        """competition モードで force=False のとき既存データファイルがあれば FileExistsError になること。"""
        (tmp_path / "train.csv").write_text("existing data")
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_cls.return_value = MagicMock()
            with pytest.raises(FileExistsError, match="force=true"):
                KaggleDownloader(competition_cfg, mock_git_repo).download()

    def test_download_succeeds_when_files_exist_and_force_true(
        self, dataset_cfg: DictConfig, mock_git_repo: MagicMock, tmp_path: Path
    ) -> None:
        """force=True のとき既存データファイルがあってもダウンロードが成功すること。"""
        (tmp_path / "train.csv").write_text("existing data")
        cfg = OmegaConf.merge(dataset_cfg, {"force": True})
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = KaggleDownloader(cfg, mock_git_repo).download()
        assert result is not None

    def test_download_succeeds_when_only_management_files_exist(
        self, dataset_cfg: DictConfig, mock_git_repo: MagicMock, tmp_path: Path
    ) -> None:
        """.gitkeep / .gitignore / metadata.yaml のみ存在する場合は force=False でもダウンロード成功すること。

        これらは git 管理用・メタデータファイルでありデータファイルではない。
        空ディレクトリと同等に扱う。
        """
        (tmp_path / ".gitkeep").write_text("")
        (tmp_path / ".gitignore").write_text("*\n!.gitkeep\n")
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = KaggleDownloader(dataset_cfg, mock_git_repo).download()
        assert result is not None


class TestKaggleDownloaderInvalidMode:
    def test_invalid_mode_raises_value_error(
        self, dataset_cfg: DictConfig, mock_git_repo: MagicMock
    ) -> None:
        """不明な mode に対して ValueError が発生すること。

        予期しない設定値に対して明示的にエラーを発生させることで、
        サイレントな不具合（無音でスキップ等）を防ぐ。
        """
        from omegaconf import DictConfig

        merged = OmegaConf.merge(dataset_cfg, {"downloader": {"mode": "unknown_mode"}})
        assert isinstance(merged, DictConfig)
        cfg = merged
        with patch("kaggle.api.kaggle_api_extended.KaggleApi") as mock_cls:
            mock_cls.return_value = MagicMock()
            downloader = KaggleDownloader(cfg, mock_git_repo)
            with pytest.raises(ValueError, match="Unknown mode"):
                downloader.download()
