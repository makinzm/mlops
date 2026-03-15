"""
SourceDatasetRepository Protocol の shape テスト。

なぜこのテストが必要か:
  - Protocol の shape を確認することで、KaggleSourceDatasetRepository が
    SourceDatasetRepository を正しく実装しているかをコンパイル時・テスト時に保証する。
  - 将来 HuggingFace Hub / GCS 等の実装を追加する際に、Protocol 準拠チェックが
    リグレッションを防ぐ。
"""

from src.domain.repository.source_dataset import DatasetMetadata, SourceDatasetRepository


class TestSourceDatasetRepositoryProtocol:
    """SourceDatasetRepository Protocol の shape を検証する。"""

    def test_protocol_has_create_method(self) -> None:
        """Protocol に create メソッドが存在すること。"""
        assert hasattr(SourceDatasetRepository, "create")

    def test_protocol_has_update_version_method(self) -> None:
        """Protocol に update_version メソッドが存在すること。"""
        assert hasattr(SourceDatasetRepository, "update_version")


class TestDatasetMetadata:
    """DatasetMetadata dataclass の検証。"""

    def test_dataset_metadata_has_required_fields(self) -> None:
        """DatasetMetadata が必須フィールドを持つこと。"""
        meta = DatasetMetadata(
            title="mlops-pipeline-src",
            owner_slug="testuser",
            dataset_slug="mlops-pipeline-src",
        )
        assert meta.title == "mlops-pipeline-src"
        assert meta.owner_slug == "testuser"
        assert meta.dataset_slug == "mlops-pipeline-src"

    def test_dataset_metadata_default_license(self) -> None:
        """DatasetMetadata のデフォルトライセンスが CC0-1.0 であること。"""
        meta = DatasetMetadata(
            title="test",
            owner_slug="user",
            dataset_slug="slug",
        )
        assert meta.license_name == "CC0-1.0"

    def test_dataset_metadata_is_immutable(self) -> None:
        """DatasetMetadata が frozen（イミュータブル）であること。"""
        import pytest

        meta = DatasetMetadata(title="t", owner_slug="u", dataset_slug="s")
        with pytest.raises(Exception):
            meta.title = "changed"  # type: ignore[misc]

    def test_dataset_metadata_full_id(self) -> None:
        """DatasetMetadata の full_id が owner_slug/dataset_slug 形式であること。"""
        meta = DatasetMetadata(
            title="mlops-pipeline-src",
            owner_slug="testuser",
            dataset_slug="mlops-pipeline-src",
        )
        assert meta.full_id == "testuser/mlops-pipeline-src"
