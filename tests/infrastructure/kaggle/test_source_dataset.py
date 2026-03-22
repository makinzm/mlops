"""
KaggleSourceDatasetRepository のテスト。

なぜこのテストが必要か:
  - create() / update_version() が Kaggle API を正しく呼び出すことを保証する。
  - dataset-metadata.json の内容が Kaggle API の要求仕様を満たすことを確認する。
  - 認証失敗（SystemExit）が RuntimeError に変換されることを確認する。
  - Kaggle API は MagicMock で差し替え、CI で実際の通信が発生しないようにする。

fixture:
  - tmp_path: staging dir の代替（実際のファイルシステムを使う）
  - MagicMock: KaggleApi の差し替え
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.domain.repository.source_dataset import DatasetMetadata
from src.infrastructure.kaggle.source_dataset import KaggleSourceDatasetRepository


def _make_metadata(owner_slug: str = "testuser") -> DatasetMetadata:
    return DatasetMetadata(
        title="mlops-pipeline-src",
        owner_slug=owner_slug,
        dataset_slug="mlops-pipeline-src",
    )


class TestKaggleSourceDatasetRepositoryCreate:
    """create() メソッドの検証。"""

    def test_create_writes_dataset_metadata_json(self, tmp_path: Path) -> None:
        """create() が staging_dir に dataset-metadata.json を書き出すこと。"""
        mock_api = MagicMock()
        repo = KaggleSourceDatasetRepository(kaggle_api=mock_api)
        meta = _make_metadata()

        repo.create(staging_dir=tmp_path, metadata=meta)

        metadata_file = tmp_path / "dataset-metadata.json"
        assert metadata_file.exists(), "dataset-metadata.json が生成されていない"
        content = json.loads(metadata_file.read_text())
        assert content["id"] == "testuser/mlops-pipeline-src"
        assert content["title"] == "mlops-pipeline-src"

    def test_create_metadata_has_licenses_field(self, tmp_path: Path) -> None:
        """create() が生成する dataset-metadata.json に licenses フィールドが含まれること。"""
        mock_api = MagicMock()
        repo = KaggleSourceDatasetRepository(kaggle_api=mock_api)
        meta = _make_metadata()

        repo.create(staging_dir=tmp_path, metadata=meta)

        content = json.loads((tmp_path / "dataset-metadata.json").read_text())
        assert "licenses" in content
        assert content["licenses"][0]["name"] == "CC0-1.0"

    def test_create_calls_kaggle_api_dataset_create_new(self, tmp_path: Path) -> None:
        """create() が KaggleApi.dataset_create_new() を1回呼ぶこと。"""
        mock_api = MagicMock()
        repo = KaggleSourceDatasetRepository(kaggle_api=mock_api)

        repo.create(staging_dir=tmp_path, metadata=_make_metadata())

        mock_api.dataset_create_new.assert_called_once()

    def test_create_passes_staging_dir_to_api(self, tmp_path: Path) -> None:
        """create() が staging_dir のパスを KaggleApi に渡すこと。"""
        mock_api = MagicMock()
        repo = KaggleSourceDatasetRepository(kaggle_api=mock_api)

        repo.create(staging_dir=tmp_path, metadata=_make_metadata())

        call_args = mock_api.dataset_create_new.call_args
        assert str(tmp_path) in str(call_args)

    def test_create_raises_runtime_error_on_auth_failure(self, tmp_path: Path) -> None:
        """create() が SystemExit を RuntimeError に変換すること。"""
        mock_api = MagicMock()
        mock_api.dataset_create_new.side_effect = SystemExit(1)
        repo = KaggleSourceDatasetRepository(kaggle_api=mock_api)

        with pytest.raises(RuntimeError, match="Kaggle"):
            repo.create(staging_dir=tmp_path, metadata=_make_metadata())


class TestKaggleSourceDatasetRepositoryUpdateVersion:
    """update_version() メソッドの検証。"""

    def test_update_version_writes_dataset_metadata_json(self, tmp_path: Path) -> None:
        """update_version() が staging_dir に dataset-metadata.json を書き出すこと。"""
        mock_api = MagicMock()
        repo = KaggleSourceDatasetRepository(kaggle_api=mock_api)

        repo.update_version(
            staging_dir=tmp_path,
            metadata=_make_metadata(),
            version_message="add target encoding",
        )

        assert (tmp_path / "dataset-metadata.json").exists()

    def test_update_version_calls_kaggle_api_create_version(self, tmp_path: Path) -> None:
        """update_version() が KaggleApi.dataset_create_version() を1回呼ぶこと。"""
        mock_api = MagicMock()
        repo = KaggleSourceDatasetRepository(kaggle_api=mock_api)

        repo.update_version(
            staging_dir=tmp_path,
            metadata=_make_metadata(),
            version_message="v2",
        )

        mock_api.dataset_create_version.assert_called_once()

    def test_update_version_passes_version_message(self, tmp_path: Path) -> None:
        """update_version() が version_message を KaggleApi に渡すこと。"""
        mock_api = MagicMock()
        repo = KaggleSourceDatasetRepository(kaggle_api=mock_api)

        repo.update_version(
            staging_dir=tmp_path,
            metadata=_make_metadata(),
            version_message="my message",
        )

        call_args = mock_api.dataset_create_version.call_args
        assert "my message" in str(call_args)

    def test_update_version_raises_runtime_error_on_auth_failure(self, tmp_path: Path) -> None:
        """update_version() が SystemExit を RuntimeError に変換すること。"""
        mock_api = MagicMock()
        mock_api.dataset_create_version.side_effect = SystemExit(1)
        repo = KaggleSourceDatasetRepository(kaggle_api=mock_api)

        with pytest.raises(RuntimeError, match="Kaggle"):
            repo.update_version(
                staging_dir=tmp_path,
                metadata=_make_metadata(),
                version_message="v2",
            )
