"""
UpdateSourceDatasetUseCase — 既存 Dataset のバージョンを更新する。

処理フロー:
1. .staging/source_dataset_{timestamp}/ を作成する
2. src/ を ignore パターンでフィルタしながらステージングにコピーする
3. SourceDatasetRepository.update_version() を呼ぶ
4. 成功したらステージングを削除する（失敗時は残す）

UseCase 層は SourceDatasetRepository Protocol にのみ依存し、プラットフォームを知らない。
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from src.domain.repository.source_dataset import DatasetMetadata, SourceDatasetRepository
from src.usecase._utils import resolve_latest_dir
from src.usecase.source_dataset._staging import (
    cleanup_staging_dir,
    copy_to_staging,
    load_ignore_patterns,
    make_staging_dir,
)

logger = logging.getLogger(__name__)


class UpdateSourceDatasetUseCase:
    """既存 Dataset の新バージョンを作成するユースケース。

    Args:
        cfg: Hydra DictConfig。以下のキーを使用する:
            - cfg.source_dataset.src_dir: アップロード対象ディレクトリ
            - cfg.source_dataset.dataset_slug: Dataset の slug
            - cfg.source_dataset.title: Dataset のタイトル
            - cfg.source_dataset.license_name: ライセンス名
            - cfg.source_dataset.ignorefile: ignore ファイルパス（省略可）
            - cfg.source_dataset.version_message: バージョンの説明
            - cfg.staging_dir: ステージングルートディレクトリ
            - cfg.platform_username: プラットフォームのユーザー名
        repository: SourceDatasetRepository を実装したオブジェクト。
    """

    def __init__(self, cfg: DictConfig, repository: SourceDatasetRepository) -> None:
        self._cfg = cfg
        self._repository = repository

    def execute(self) -> None:
        """src/ と conf/ を Dataset の新バージョンとしてアップロードする。

        Raises:
            RuntimeError: repository.update_version() が失敗した場合（ステージングを残す）。
        """
        src_dir = resolve_latest_dir(str(self._cfg.source_dataset.src_dir))
        conf_dir_raw = self._cfg.source_dataset.get("conf_dir")
        conf_dir = (
            resolve_latest_dir(str(conf_dir_raw)) if conf_dir_raw else src_dir.parent / "conf"
        )
        staging_root = Path(str(self._cfg.staging_dir))
        ignore_file_raw = self._cfg.source_dataset.get("ignorefile")
        ignore_file_path = Path(str(ignore_file_raw)) if ignore_file_raw else None
        version_message: str = str(
            self._cfg.source_dataset.get("version_message", "update source code")
        )

        metadata = DatasetMetadata(
            title=str(self._cfg.source_dataset.title),
            owner_slug=str(self._cfg.get("platform_username") or ""),
            dataset_slug=str(self._cfg.source_dataset.dataset_slug),
            license_name=str(self._cfg.source_dataset.get("license_name", "CC0-1.0")),
        )

        patterns = load_ignore_patterns(ignore_file_path)
        staging_dir = make_staging_dir(staging_root)
        requirements_path = src_dir.parent / "requirements.txt"

        # extra_dirs: モデルディレクトリなど追加でアップロードするディレクトリ
        extra_dirs_raw = self._cfg.source_dataset.get("extra_dirs", [])
        extra_dirs = [Path(str(d)) for d in extra_dirs_raw] if extra_dirs_raw else []

        logger.info("Staging src/ and conf/ to %s", staging_dir)
        copy_to_staging(src_dir, conf_dir, staging_dir, patterns, requirements_path, extra_dirs)

        logger.info("Updating Dataset: %s (version: %s)", metadata.full_id, version_message)
        try:
            self._repository.update_version(
                staging_dir=staging_dir,
                metadata=metadata,
                version_message=version_message,
            )
        except Exception as update_error:
            # Dataset が存在しない場合（403 等）は自動で create にフォールバック
            logger.warning("Dataset バージョン更新に失敗。新規作成を試みます: %s", metadata.full_id)
            try:
                self._repository.create(staging_dir=staging_dir, metadata=metadata)
                logger.info("Dataset を新規作成しました: %s", metadata.full_id)
            except Exception:
                logger.error(
                    "Dataset 新規作成にも失敗しました。staging dir は残しています: %s",
                    staging_dir,
                )
                raise update_error

        cleanup_staging_dir(staging_dir)
        logger.info("Done. Dataset: %s", metadata.full_id)
