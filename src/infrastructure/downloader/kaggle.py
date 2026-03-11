"""
Kaggle API を使ったデータダウンロード実装。

認証方式（優先順）:
  1. ~/.kaggle/access_token（推奨: トークン文字列のみ記載）
  2. KAGGLE_API_TOKEN 環境変数
  3. ~/.kaggle/kaggle.json（Legacy）
"""

from pathlib import Path

from omegaconf import DictConfig

from src.domain.data.downloader import DownloadResult
from src.domain.repository.git import GitRepository


class KaggleDownloader:
    def __init__(self, cfg: DictConfig, git_repo: GitRepository) -> None:
        from kaggle.api.kaggle_api_extended import (  # type: ignore[import-untyped]
            KaggleApi as KaggleApiExtended,
        )

        self.cfg = cfg
        self.git_repo = git_repo
        self.api = KaggleApiExtended()
        self.api.authenticate()

    def download(self) -> DownloadResult:
        mode = self.cfg.downloader.mode
        if mode == "dataset":
            return self._download_dataset()
        elif mode == "competition":
            return self._download_competition()
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Choose 'dataset' or 'competition'.")

    def _download_dataset(self) -> DownloadResult:
        output_dir = Path(self.cfg.output_dir)
        self.git_repo.setup_data_dir(output_dir)
        self.api.dataset_download_files(
            dataset=self.cfg.downloader.dataset,
            path=output_dir,
            unzip=self.cfg.unzip,
            force=self.cfg.force,
        )
        files = [f for f in output_dir.rglob("*") if f.is_file() and f.name != ".gitkeep"]
        return DownloadResult(
            output_dir=output_dir,
            files=files,
            commit_hash=self.git_repo.get_commit_hash(),
        )

    def _download_competition(self) -> DownloadResult:
        output_dir = Path(self.cfg.output_dir)
        self.git_repo.setup_data_dir(output_dir)
        self.api.competition_download_files(
            competition=self.cfg.downloader.competition,
            path=output_dir,
            force=self.cfg.force,
        )
        files = [f for f in output_dir.rglob("*") if f.is_file() and f.name != ".gitkeep"]
        return DownloadResult(
            output_dir=output_dir,
            files=files,
            commit_hash=self.git_repo.get_commit_hash(),
        )
