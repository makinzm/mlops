"""
ImageResolver のテスト。

なぜこのテストが必要か:
  - ImageResolver が画像の存在確認・メタデータ取得を行うこと。
  - validate_images メソッドが存在しない画像パスを検出すること。
  - create_image_metadata メソッドが画像の幅・高さ・チャネル数を返すこと。
  - 既存の Resolver パターン（PolarsResolver, SklearnResolver）と同じインターフェースに従うこと。

時間計算量: O(N) — N: 行数
空間計算量: O(N)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

from src.infrastructure.preprocessor.resolvers.image_resolver import ImageResolver


def _create_test_images(tmp_path: Path, sizes: list[tuple[int, int]]) -> list[str]:
    """テスト用画像を作成して Path リストを返す。"""
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    rng = np.random.default_rng(42)
    for i, (w, h) in enumerate(sizes):
        data = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        path = image_dir / f"img_{i}.png"
        Image.fromarray(data).save(path)
        paths.append(str(path))
    return paths


class TestImageResolver:
    def test_supported_methods(self) -> None:
        """supported_methods が validate_images と create_image_metadata を返すこと。"""
        resolver = ImageResolver()
        methods = resolver.supported_methods()
        assert "validate_images" in methods
        assert "create_image_metadata" in methods

    def test_validate_images_all_exist(self, tmp_path: Path) -> None:
        """全画像が存在する場合、__image_valid__ が全て True であること。"""
        paths = _create_test_images(tmp_path, [(32, 32), (64, 64)])
        df = pl.DataFrame({"image_path": paths})
        resolver = ImageResolver()
        result = resolver.execute(df, "validate_images", column="image_path")
        assert "__image_valid__" in result.columns
        assert result["__image_valid__"].to_list() == [True, True]

    def test_validate_images_missing_file(self, tmp_path: Path) -> None:
        """存在しない画像パスがある場合、__image_valid__ が False になること。"""
        paths = _create_test_images(tmp_path, [(32, 32)])
        paths.append("/nonexistent/image.png")
        df = pl.DataFrame({"image_path": paths})
        resolver = ImageResolver()
        result = resolver.execute(df, "validate_images", column="image_path")
        assert result["__image_valid__"].to_list() == [True, False]

    def test_create_image_metadata(self, tmp_path: Path) -> None:
        """画像メタデータが正しく取得されること。"""
        paths = _create_test_images(tmp_path, [(32, 48), (64, 64)])
        df = pl.DataFrame({"image_path": paths})
        resolver = ImageResolver()
        result = resolver.execute(df, "create_image_metadata", column="image_path")
        assert "__image_width__" in result.columns
        assert "__image_height__" in result.columns
        assert "__image_channels__" in result.columns
        assert result["__image_width__"].to_list() == [32, 64]
        assert result["__image_height__"].to_list() == [48, 64]
        assert result["__image_channels__"].to_list() == [3, 3]
