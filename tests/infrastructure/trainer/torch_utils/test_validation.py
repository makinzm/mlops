"""
torch_utils/validation のテスト。

なぜこのテストが必要か:
  - validate_training_inputs() が学習前に入力データを検証すること。
  - 画像パスが存在しない場合にエラーを検出すること。
  - ラベルが num_classes の範囲外の場合にエラーを検出すること。
  - 画像サイズとモデル期待サイズの不一致を警告すること。
  - エラーメッセージに修正方法が含まれること。

時間計算量: O(N) — N: サンプル数（サンプルチェック）
空間計算量: O(1)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

from src.domain.model.backbone import BackboneConfig
from src.infrastructure.trainer.torch_utils.validation import (
    validate_training_inputs,
)


def _create_images(image_dir: Path, num: int, size: int = 64) -> list[str]:
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    paths: list[str] = []
    for i in range(num):
        data = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        path = image_dir / f"img_{i}.png"
        Image.fromarray(data).save(path)
        paths.append(str(path))
    return paths


def _make_fold_data(
    tmp_path: Path,
    image_size: int = 64,
    num_train: int = 10,
    num_valid: int = 5,
    labels_range: tuple[int, int] = (0, 2),
) -> Path:
    """fold_0/ に train.parquet / test.parquet を作成する。"""
    fold_dir = tmp_path / "preprocess" / "fold_0"
    fold_dir.mkdir(parents=True)
    rng = np.random.default_rng(42)

    train_paths = _create_images(tmp_path / "images" / "train", num_train, image_size)
    valid_paths = _create_images(tmp_path / "images" / "valid", num_valid, image_size)

    pl.DataFrame(
        {
            "image_path": train_paths,
            "label": rng.integers(labels_range[0], labels_range[1], num_train).tolist(),
        }
    ).write_parquet(fold_dir / "train.parquet")

    pl.DataFrame(
        {
            "image_path": valid_paths,
            "label": rng.integers(labels_range[0], labels_range[1], num_valid).tolist(),
        }
    ).write_parquet(fold_dir / "test.parquet")

    return tmp_path / "preprocess"


class TestValidateTrainingInputs:
    def test_valid_inputs_return_no_errors(self, tmp_path: Path) -> None:
        """正しい入力ではエラーが返らないこと。"""
        preprocess_dir = _make_fold_data(tmp_path)
        config = BackboneConfig(
            backbone_name="simple_cnn",
            num_classes=2,
            pretrained=False,
            image_size=32,
        )
        issues = validate_training_inputs(
            preprocess_output_dir=preprocess_dir,
            backbone_config=config,
            target_col="label",
            image_path_col="image_path",
            num_classes=2,
        )
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0

    def test_missing_fold_directory(self, tmp_path: Path) -> None:
        """fold ディレクトリが存在しない場合にエラーを検出すること。"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        config = BackboneConfig(backbone_name="simple_cnn", num_classes=2, pretrained=False)
        issues = validate_training_inputs(
            preprocess_output_dir=empty_dir,
            backbone_config=config,
            target_col="label",
            image_path_col="image_path",
            num_classes=2,
        )
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) > 0
        assert any("fold" in i.message.lower() for i in errors)

    def test_missing_image_path_detected(self, tmp_path: Path) -> None:
        """存在しない画像パスがある場合にエラーを検出すること。"""
        fold_dir = tmp_path / "preprocess" / "fold_0"
        fold_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "image_path": ["/nonexistent/image.png"],
                "label": [0],
            }
        ).write_parquet(fold_dir / "train.parquet")
        pl.DataFrame(
            {
                "image_path": ["/nonexistent/image2.png"],
                "label": [1],
            }
        ).write_parquet(fold_dir / "test.parquet")

        config = BackboneConfig(backbone_name="simple_cnn", num_classes=2, pretrained=False)
        issues = validate_training_inputs(
            preprocess_output_dir=tmp_path / "preprocess",
            backbone_config=config,
            target_col="label",
            image_path_col="image_path",
            num_classes=2,
        )
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) > 0
        assert any("image" in i.message.lower() or "exist" in i.message.lower() for i in errors)

    def test_label_out_of_range_detected(self, tmp_path: Path) -> None:
        """ラベルが num_classes の範囲外の場合にエラーを検出すること。"""
        preprocess_dir = _make_fold_data(tmp_path, labels_range=(0, 5))
        config = BackboneConfig(backbone_name="simple_cnn", num_classes=2, pretrained=False)
        issues = validate_training_inputs(
            preprocess_output_dir=preprocess_dir,
            backbone_config=config,
            target_col="label",
            image_path_col="image_path",
            num_classes=2,
        )
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) > 0
        assert any("label" in i.message.lower() or "class" in i.message.lower() for i in errors)

    def test_image_size_mismatch_warning(self, tmp_path: Path) -> None:
        """画像サイズとモデル期待サイズが大きく異なる場合に警告すること。"""
        preprocess_dir = _make_fold_data(tmp_path, image_size=512)
        config = BackboneConfig(
            backbone_name="simple_cnn",
            num_classes=2,
            pretrained=False,
            image_size=32,
        )
        issues = validate_training_inputs(
            preprocess_output_dir=preprocess_dir,
            backbone_config=config,
            target_col="label",
            image_path_col="image_path",
            num_classes=2,
        )
        warnings = [i for i in issues if i.severity == "warning"]
        assert len(warnings) > 0
        assert any("size" in i.message.lower() or "resize" in i.message.lower() for i in warnings)

    def test_error_message_contains_fix_suggestion(self, tmp_path: Path) -> None:
        """エラーメッセージに修正方法が含まれること。"""
        preprocess_dir = _make_fold_data(tmp_path, image_size=512)
        config = BackboneConfig(
            backbone_name="simple_cnn",
            num_classes=2,
            pretrained=False,
            image_size=32,
        )
        issues = validate_training_inputs(
            preprocess_output_dir=preprocess_dir,
            backbone_config=config,
            target_col="label",
            image_path_col="image_path",
            num_classes=2,
        )
        warnings = [i for i in issues if i.severity == "warning"]
        assert any("backbone.image_size" in i.message for i in warnings)
