"""
GCSRepositoryImpl — Google Cloud Storage を使った GCSRepository の実装。

upload_dir: ローカルディレクトリを GCS にアップロードする。
download_dir: GCS プレフィックス以下のファイルをローカルにダウンロードする。
"""

from __future__ import annotations

import logging
from pathlib import Path

from google.cloud import storage

logger = logging.getLogger(__name__)


def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """gs://bucket/prefix → (bucket, prefix) に分解する。"""
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri!r}. Must start with 'gs://'")
    path = gcs_uri[5:]
    parts = path.split("/", 1)
    bucket = parts[0]
    prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
    return bucket, prefix


class GCSRepositoryImpl:
    """GCSRepository の google-cloud-storage による実装。"""

    def __init__(self, project: str) -> None:
        self._client = storage.Client(project=project)

    def upload_dir(self, local_dir: Path, gcs_uri: str) -> None:
        """ローカルディレクトリを GCS にアップロードする。

        local_dir 直下の全ファイル（再帰）を gcs_uri/relative_path に配置する。
        """
        bucket_name, prefix = _parse_gcs_uri(gcs_uri)
        bucket = self._client.bucket(bucket_name)

        files = sorted(p for p in local_dir.rglob("*") if p.is_file())
        logger.info(f"Uploading {len(files)} files to {gcs_uri}")
        for local_file in files:
            relative = local_file.relative_to(local_dir)
            blob_name = f"{prefix}/{relative}" if prefix else str(relative)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(local_file))
            logger.debug(f"Uploaded: {local_file} → gs://{bucket_name}/{blob_name}")

    def download_dir(self, gcs_uri: str, local_dir: Path) -> None:
        """GCS プレフィックス以下の全ファイルをローカルにダウンロードする。

        GCS の blob.name から prefix を除いた相対パスを local_dir 以下に配置する。
        """
        bucket_name, prefix = _parse_gcs_uri(gcs_uri)
        blobs = list(self._client.list_blobs(bucket_name, prefix=prefix))
        logger.info(f"Downloading {len(blobs)} files from {gcs_uri} to {local_dir}")
        for blob in blobs:
            # prefix の後の相対パス部分だけを取り出す
            relative_str = blob.name[len(prefix) :].lstrip("/") if prefix else blob.name
            if not relative_str:
                continue  # prefix 自体を指す blob は無視
            local_file = local_dir / relative_str
            local_file.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_file))
            logger.debug(f"Downloaded: gs://{bucket_name}/{blob.name} → {local_file}")
