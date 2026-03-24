"""
Phase 1: BackboneConfig / DimensionInfo dataclass のテスト。

なぜこのテストが必要か:
  - BackboneConfig は backbone 名・pretrained フラグ・入力画像サイズ・クラス数を保持する。
    これらのフィールドが正しくインスタンス化されることを確認する。
  - DimensionInfo は次元ミスマッチ検出結果を保持する。backbone_name, output_features,
    expected_features, message, is_valid が正しく設定されることを確認する。
  - Domain 層のため torch/torchvision に依存してはならない。
"""

from __future__ import annotations

from src.domain.model.backbone import BackboneConfig, DimensionInfo


class TestBackboneConfig:
    def test_create_with_defaults(self) -> None:
        """デフォルト値でインスタンス化できること。"""
        config = BackboneConfig(
            backbone_name="resnet50",
            num_classes=10,
        )
        assert config.backbone_name == "resnet50"
        assert config.num_classes == 10
        assert config.pretrained is True
        assert config.image_size == 224

    def test_create_with_custom_values(self) -> None:
        """カスタム値を設定できること。"""
        config = BackboneConfig(
            backbone_name="vit_b_16",
            num_classes=2,
            pretrained=False,
            image_size=384,
        )
        assert config.backbone_name == "vit_b_16"
        assert config.num_classes == 2
        assert config.pretrained is False
        assert config.image_size == 384


class TestDimensionInfo:
    def test_valid_dimensions(self) -> None:
        """次元が一致する場合 is_valid=True であること。"""
        info = DimensionInfo(
            backbone_name="resnet50",
            output_features=2048,
            expected_features=2048,
            message="OK",
            is_valid=True,
        )
        assert info.is_valid is True
        assert info.output_features == info.expected_features

    def test_invalid_dimensions(self) -> None:
        """次元が不一致の場合 is_valid=False であること。"""
        info = DimensionInfo(
            backbone_name="resnet50",
            output_features=512,
            expected_features=2048,
            message="Dimension mismatch: expected 2048, got 512",
            is_valid=False,
        )
        assert info.is_valid is False
        assert info.output_features != info.expected_features
