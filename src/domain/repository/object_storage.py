"""
オブジェクトストレージリポジトリのドメイン定義。

なぜここに定義するか:
  UseCase 層はオブジェクトストレージの具体実装を知らない。Protocol を通じてのみ依存する。
  これにより infrastructure を差し替えやすくなる。
"""

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ObjectStorageRepository(Protocol):
    """オブジェクトストレージ操作の抽象 Protocol。"""

    def upload_dir(self, local_dir: Path, remote_uri: str) -> None:
        """ローカルディレクトリをオブジェクトストレージにアップロードする。

        Args:
            local_dir: アップロード元ローカルディレクトリ。
            remote_uri: アップロード先 URI（例: gs://bucket/prefix）。
        """
        ...

    def download_dir(self, remote_uri: str, local_dir: Path) -> None:
        """オブジェクトストレージから指定プレフィックスのファイルをローカルにダウンロードする。

        Args:
            remote_uri: ダウンロード元 URI（例: gs://bucket/prefix）。
            local_dir: ダウンロード先ローカルディレクトリ。
        """
        ...
