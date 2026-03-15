"""
CreateSourceDatasetUseCase — src/ を Kaggle Dataset として新規作成する。

処理フロー:
1. .staging/source_dataset_{timestamp}/ を作成する
2. src/ を .kaggleignore でフィルタしながらステージングにコピーする
3. SourceDatasetRepository.create() を呼ぶ
4. 成功したらステージングを削除する（失敗時は残す）

UseCase 層は SourceDatasetRepository Protocol にのみ依存し、Kaggle を知らない。
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from src.domain.repository.source_dataset import DatasetMetadata, SourceDatasetRepository
from src.usecase.source_dataset._staging import (
    cleanup_staging_dir,
    copy_src_to_staging,
    load_kaggleignore_patterns,
    make_staging_dir,
)

logger = logging.getLogger(__name__)


class CreateSourceDatasetUseCase:
    """src/ を Kaggle Dataset として新規作成するユースケース。

    Args:
        cfg: Hydra DictConfig。以下のキーを使用する:
            - cfg.source_dataset.src_dir: アップロード対象ディレクトリ
            - cfg.source_dataset.dataset_slug: Dataset の slug
            - cfg.source_dataset.title: Dataset のタイトル
            - cfg.source_dataset.license_name: ライセンス名
            - cfg.source_dataset.kaggleignore: .kaggleignore ファイルパス（省略可）
            - cfg.staging_dir: ステージングルートディレクトリ
            - cfg.kaggle_username: Kaggle ユーザー名
        repository: SourceDatasetRepository を実装したオブジェクト。
    """

    def __init__(self, cfg: DictConfig, repository: SourceDatasetRepository) -> None:
        self._cfg = cfg
        self._repository = repository

    def execute(self) -> None:
        """src/ を Kaggle Dataset として新規作成する。

        Raises:
            RuntimeError: repository.create() が失敗した場合（ステージングを残す）。
        """
        src_dir = Path(str(self._cfg.source_dataset.src_dir))
        staging_root = Path(str(self._cfg.staging_dir))
        kaggleignore_raw = self._cfg.source_dataset.get("kaggleignore")
        kaggleignore_path = Path(str(kaggleignore_raw)) if kaggleignore_raw else None

        metadata = DatasetMetadata(
            title=str(self._cfg.source_dataset.title),
            owner_slug=str(self._cfg.get("kaggle_username") or ""),
            dataset_slug=str(self._cfg.source_dataset.dataset_slug),
            license_name=str(self._cfg.source_dataset.get("license_name", "CC0-1.0")),
        )

        patterns = load_kaggleignore_patterns(kaggleignore_path)
        staging_dir = make_staging_dir(staging_root)

        logger.info("Staging src/ to %s", staging_dir)
        copy_src_to_staging(src_dir, staging_dir, patterns)

        logger.info("Creating Kaggle Dataset: %s", metadata.full_id)
        # 失敗時はステージングを残すため try/except で囲む
        try:
            self._repository.create(staging_dir=staging_dir, metadata=metadata)
        except Exception:
            logger.error("Dataset 作成に失敗しました。staging dir は残しています: %s", staging_dir)
            raise

        cleanup_staging_dir(staging_dir)
        logger.info("Done. Dataset: %s", metadata.full_id)
