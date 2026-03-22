"""
GCSRepositoryImpl の単体テスト。

なぜこのテストが必要か:
  - GCSRepositoryImpl は google-cloud-storage を使い実際の GCS と通信する。
  - テストでは google.cloud.storage.Client をモックし、実際の GCS 呼び出しを防ぐ。
  - upload_dir が全ファイルを正しいパスで GCS に送ること、
    download_dir が全 blob をローカルに展開することを保証する。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.gcp.storage import GCSRepositoryImpl


class TestGCSRepositoryImplUploadDir:
    """upload_dir のテスト。"""

    def test_uploads_all_files_in_dir(self, tmp_path: Path) -> None:
        """ディレクトリ内の全ファイルが正しい GCS パスにアップロードされること。"""
        # Arrange
        (tmp_path / "fold_0").mkdir()
        (tmp_path / "fold_0" / "model.lgbm").write_bytes(b"model_data")
        (tmp_path / "fold_0" / "oof_train.parquet").write_bytes(b"parquet_data")
        (tmp_path / "train_result.yaml").write_text("cv_mean_score: 0.85")

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        with patch("src.infrastructure.gcp.storage.storage") as mock_storage_module:
            mock_storage_module.Client.return_value = mock_client
            repo = GCSRepositoryImpl(project="test-project")
            repo.upload_dir(tmp_path, "gs://test-bucket/staging/job1/models")

        # Assert: 3 ファイルがアップロードされている
        assert mock_blob.upload_from_filename.call_count == 3

    def test_upload_preserves_relative_paths(self, tmp_path: Path) -> None:
        """サブディレクトリ構造が GCS の blob パスに反映されること。"""
        subdir = tmp_path / "fold_0"
        subdir.mkdir()
        (subdir / "model.lgbm").write_bytes(b"data")

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        with patch("src.infrastructure.gcp.storage.storage") as mock_storage_module:
            mock_storage_module.Client.return_value = mock_client
            repo = GCSRepositoryImpl(project="test-project")
            repo.upload_dir(tmp_path, "gs://test-bucket/prefix")

        # blob パスが prefix/fold_0/model.lgbm になること
        called_paths = [call.args[0] for call in mock_bucket.blob.call_args_list]
        assert any("fold_0/model.lgbm" in p for p in called_paths)


class TestGCSRepositoryImplDownloadDir:
    """download_dir のテスト。"""

    def test_downloads_all_blobs(self, tmp_path: Path) -> None:
        """GCS プレフィックス以下の全 blob がローカルにダウンロードされること。"""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        # 2 つの blob をシミュレート
        blob1 = MagicMock()
        blob1.name = "staging/job1/models/train_result.yaml"
        blob2 = MagicMock()
        blob2.name = "staging/job1/models/fold_0/model.lgbm"
        mock_client.list_blobs.return_value = [blob1, blob2]

        with patch("src.infrastructure.gcp.storage.storage") as mock_storage_module:
            mock_storage_module.Client.return_value = mock_client
            repo = GCSRepositoryImpl(project="test-project")
            repo.download_dir("gs://test-bucket/staging/job1/models", tmp_path)

        # 2 ファイルがダウンロードされている
        assert blob1.download_to_filename.call_count == 1
        assert blob2.download_to_filename.call_count == 1

    def test_download_creates_subdirectories(self, tmp_path: Path) -> None:
        """サブディレクトリが存在しなくても自動作成されること。"""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        blob = MagicMock()
        blob.name = "prefix/fold_0/model.lgbm"

        def fake_download(filename: str) -> None:
            Path(filename).write_bytes(b"data")

        blob.download_to_filename.side_effect = fake_download
        mock_client.list_blobs.return_value = [blob]

        with patch("src.infrastructure.gcp.storage.storage") as mock_storage_module:
            mock_storage_module.Client.return_value = mock_client
            repo = GCSRepositoryImpl(project="test-project")
            repo.download_dir("gs://test-bucket/prefix", tmp_path)

        assert (tmp_path / "fold_0" / "model.lgbm").exists()
