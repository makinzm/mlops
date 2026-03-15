"""
Kaggle API を使ったデータダウンロード実装。

認証方式（優先順）:
  1. ~/.kaggle/access_token（推奨: トークン文字列のみ記載）
  2. KAGGLE_API_TOKEN 環境変数
  3. ~/.kaggle/kaggle.json（Legacy）
"""

import zipfile
from datetime import datetime
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

from src.domain.data.downloader import DownloadResult
from src.domain.repository.git import GitRepository

_METADATA_EXCLUDE = {".gitkeep", ".gitignore", "metadata.yaml"}


class KaggleDownloader:
    def __init__(self, cfg: DictConfig, git_repo: GitRepository) -> None:
        # kaggle/__init__.py が import 時に api.authenticate() を実行するため、
        # SystemExit はここで捕捉する必要がある。
        try:
            from kaggle.api.kaggle_api_extended import (  # type: ignore[import-untyped]
                KaggleApi as KaggleApiExtended,
            )
        except SystemExit as e:
            raise RuntimeError(
                "Kaggle 認証に失敗しました。~/.kaggle/access_token を確認してください。"
            ) from e

        self.cfg = cfg
        self.git_repo = git_repo
        self.api = KaggleApiExtended()
        try:
            self.api.authenticate()
        except SystemExit as e:
            raise RuntimeError(
                "Kaggle 認証に失敗しました。~/.kaggle/access_token を確認してください。"
            ) from e

    def download(self) -> DownloadResult:
        mode = self.cfg.kaggle.mode
        if mode == "dataset":
            return self._download_dataset()
        elif mode == "competition":
            return self._download_competition()
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Choose 'dataset' or 'competition'.")

    def _download_dataset(self) -> DownloadResult:
        if not self.cfg.kaggle.dataset:
            raise ValueError(
                "dataset が未指定です。 例: uv run python -m src kaggle.dataset=owner/dataset-name"
            )
        output_dir = Path(self.cfg.output_dir)
        self.git_repo.setup_data_dir(output_dir)
        self._check_force(output_dir)
        self.api.dataset_download_files(
            dataset=self.cfg.kaggle.dataset,
            path=output_dir,
            unzip=False,
            force=self.cfg.force,
        )
        if self.cfg.unzip:
            self._unzip_zips(output_dir)
        return self._build_result(output_dir)

    def _download_competition(self) -> DownloadResult:
        if not self.cfg.kaggle.competition:
            raise ValueError(
                "competition が未指定です。 例: uv run python -m src kaggle.competition=titanic"
            )
        output_dir = Path(self.cfg.output_dir)
        self.git_repo.setup_data_dir(output_dir)
        self._check_force(output_dir)
        self.api.competition_download_files(
            competition=self.cfg.kaggle.competition,
            path=output_dir,
            force=self.cfg.force,
        )
        if self.cfg.unzip:
            self._unzip_zips(output_dir)
        return self._build_result(output_dir)

    def _check_force(self, output_dir: Path) -> None:
        """force=False かつデータファイルが存在する場合は FileExistsError を送出する。"""
        if not self.cfg.force:
            existing = [
                f for f in output_dir.rglob("*") if f.is_file() and f.name not in _METADATA_EXCLUDE
            ]
            if existing:
                raise FileExistsError(
                    f"'{output_dir}' にデータファイルが存在します。"
                    " 再ダウンロードするには force=true を指定してください。"
                )

    def _unzip_zips(self, output_dir: Path) -> None:
        """output_dir 内の ZIP ファイルをすべて展開し、ZIP を削除する。"""
        for zip_path in list(output_dir.rglob("*.zip")):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(output_dir)
            zip_path.unlink()

    def _build_result(self, output_dir: Path) -> DownloadResult:
        files = [
            f for f in output_dir.rglob("*") if f.is_file() and f.name not in _METADATA_EXCLUDE
        ]
        commit_hash = self.git_repo.get_commit_hash()
        self._save_metadata(output_dir, files, commit_hash)
        return DownloadResult(output_dir=output_dir, files=files, commit_hash=commit_hash)

    def _save_metadata(self, output_dir: Path, files: list[Path], commit_hash: str) -> None:
        metadata = {
            "downloaded_at": datetime.now().isoformat(timespec="seconds"),
            "commit_hash": commit_hash,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "files": [f.name for f in files],
        }
        (output_dir / "metadata.yaml").write_text(
            yaml.dump(metadata, default_flow_style=False, allow_unicode=True, sort_keys=False)
        )
