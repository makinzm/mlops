"""
GCS (Google Cloud Storage) リポジトリのドメイン定義。

なぜここに定義するか:
  UseCase 層は GCS の具体実装を知らない。Protocol を通じてのみ依存する。
  これにより infrastructure を差し替えやすくなる。
"""

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class GCSRepository(Protocol):
    """GCS オブジェクトストレージ操作の抽象 Protocol。"""

    def upload_dir(self, local_dir: Path, gcs_uri: str) -> None:
        """ローカルディレクトリを GCS にアップロードする。

        Args:
            local_dir: アップロード元ローカルディレクトリ。
            gcs_uri: アップロード先 GCS URI（例: gs://bucket/prefix）。
        """
        ...

    def download_dir(self, gcs_uri: str, local_dir: Path) -> None:
        """GCS から指定プレフィックスのファイルをローカルにダウンロードする。

        Args:
            gcs_uri: ダウンロード元 GCS URI（例: gs://bucket/prefix）。
            local_dir: ダウンロード先ローカルディレクトリ。
        """
        ...
