"""
Kaggle API を使ったデータダウンロード実装。

新認証方式（KAGGLE_API_TOKEN 環境変数 / ~/.kaggle/access_token）を優先し、
旧方式（~/.kaggle/kaggle.json）にもフォールバックする。
"""

import subprocess
from pathlib import Path

from omegaconf import DictConfig

from src.domain.data.downloader import DownloadResult


class KaggleDownloader:
    def __init__(self, cfg: DictConfig) -> None:
        from kaggle.api.kaggle_api_extended import (  # type: ignore[import-untyped]
            KaggleApi as KaggleApiExtended,
        )

        self.cfg = cfg
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
        output_dir.mkdir(parents=True, exist_ok=True)
        self.api.dataset_download_files(
            dataset=self.cfg.downloader.dataset,
            path=output_dir,
            unzip=self.cfg.unzip,
            force=self.cfg.force,
        )
        files = list(output_dir.rglob("*")) if output_dir.exists() else []
        return DownloadResult(
            output_dir=output_dir,
            files=[f for f in files if f.is_file()],
            commit_hash=self._get_commit_hash(),
        )

    def _download_competition(self) -> DownloadResult:
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.api.competition_download_files(
            competition=self.cfg.downloader.competition,
            path=output_dir,
            force=self.cfg.force,
        )
        files = list(output_dir.rglob("*")) if output_dir.exists() else []
        return DownloadResult(
            output_dir=output_dir,
            files=[f for f in files if f.is_file()],
            commit_hash=self._get_commit_hash(),
        )

    def _get_commit_hash(self) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
