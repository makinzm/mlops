"""
SourceDatasetRepository — ソースコード Dataset のリポジトリ操作プロトコル。

プラットフォーム固有の実装を usecase 層から隠蔽し、
将来別のプラットフォームへの差し替えをこの Protocol を実装するだけで対応できるようにする。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class DatasetMetadata:
    """Dataset のメタデータ。

    Attributes:
        title: Dataset のタイトル（プラットフォーム上での表示名）。
        owner_slug: Dataset を所有するユーザー名。
        dataset_slug: Dataset の URL slug（例: "mlops-pipeline-src"）。
        license_name: ライセンス名（デフォルト: CC0-1.0）。
    """

    title: str
    owner_slug: str
    dataset_slug: str
    license_name: str = "CC0-1.0"

    @property
    def full_id(self) -> str:
        """Dataset の完全 ID（owner_slug/dataset_slug 形式）を返す。"""
        return f"{self.owner_slug}/{self.dataset_slug}"


@runtime_checkable
class SourceDatasetRepository(Protocol):
    """ソースコードの Dataset リポジトリ操作の抽象インターフェース。

    UseCase 層はこの Protocol にのみ依存し、
    具体的なプラットフォーム実装を知らない。
    """

    def create(self, staging_dir: Path, metadata: DatasetMetadata) -> None:
        """Dataset を新規作成する。

        Args:
            staging_dir: アップロードするファイルが配置されたディレクトリ。
            metadata: Dataset のメタデータ（タイトル・slug・ライセンス）。
        """
        ...

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
            version_message: バージョンの説明（変更内容のサマリー）。
        """
        ...
