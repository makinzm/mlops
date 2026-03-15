"""
KaggleSourceDatasetRepository — Kaggle API を使った SourceDatasetRepository の実装。

処理フロー（create）:
  1. staging_dir/ に dataset-metadata.json を書き出す
  2. KaggleApi.dataset_create_new(staging_dir) を呼ぶ

処理フロー（update_version）:
  1. staging_dir/ に dataset-metadata.json を書き出す
  2. KaggleApi.dataset_create_version(staging_dir, version_message) を呼ぶ

caution.md: kaggle/__init__.py は import 時に api.authenticate() を実行するため、
            SystemExit はここで捕捉して RuntimeError に変換する。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.domain.repository.source_dataset import DatasetMetadata

logger = logging.getLogger(__name__)


class KaggleSourceDatasetRepository:
    """KaggleApi を使った SourceDatasetRepository の実装。

    Args:
        kaggle_api: KaggleApi インスタンス（認証済み）。
                    テスト時は MagicMock で差し替える。
    """

    def __init__(self, kaggle_api: Any) -> None:
        self._api = kaggle_api

    def create(self, staging_dir: Path, metadata: DatasetMetadata) -> None:
        """Dataset を新規作成する。

        Args:
            staging_dir: アップロードするファイルが配置されたディレクトリ。
            metadata: Dataset のメタデータ。

        Raises:
            RuntimeError: Kaggle API が SystemExit を上げた場合（認証失敗等）。
        """
        self._write_metadata_json(staging_dir, metadata)
        try:
            self._api.dataset_create_new(str(staging_dir), public=False, quiet=False)
            logger.info("Kaggle Dataset created: %s", metadata.full_id)
        except SystemExit as e:
            raise RuntimeError(
                f"Kaggle API の呼び出しに失敗しました（dataset_create_new）: {metadata.full_id}"
            ) from e

    def update_version(
        self,
        staging_dir: Path,
        metadata: DatasetMetadata,
        version_message: str,
    ) -> None:
        """既存 Dataset の新バージョンを作成する。

        Args:
            staging_dir: アップロードするファイルが配置されたディレクトリ。
            metadata: Dataset のメタデータ。
            version_message: バージョンの説明。

        Raises:
            RuntimeError: Kaggle API が SystemExit を上げた場合（認証失敗等）。
        """
        self._write_metadata_json(staging_dir, metadata)
        try:
            self._api.dataset_create_version(
                str(staging_dir),
                version_notes=version_message,
                quiet=False,
                convert_to_csv=False,
            )
            logger.info(
                "Kaggle Dataset version created: %s (%s)", metadata.full_id, version_message
            )
        except SystemExit as e:
            raise RuntimeError(
                f"Kaggle API の呼び出しに失敗しました（dataset_create_version）: {metadata.full_id}"
            ) from e

    @staticmethod
    def _write_metadata_json(staging_dir: Path, metadata: DatasetMetadata) -> None:
        """Kaggle Dataset API が要求する dataset-metadata.json を staging_dir に書き出す。"""
        content = {
            "title": metadata.title,
            "id": metadata.full_id,
            "licenses": [{"name": metadata.license_name}],
        }
        metadata_path = staging_dir / "dataset-metadata.json"
        metadata_path.write_text(json.dumps(content, ensure_ascii=False, indent=2))
        logger.info("dataset-metadata.json written: %s", metadata_path)
