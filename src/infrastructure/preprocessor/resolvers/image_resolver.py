"""
画像前処理 Resolver。

validate_images: 画像パスの存在確認。
create_image_metadata: 画像の幅・高さ・チャネル数を取得。

既存の PolarsResolver / SklearnResolver と同じインターフェースに従う。

時間計算量: O(N) — N: 行数
空間計算量: O(N)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from PIL import Image


class ImageResolver:
    """画像データの前処理 Resolver。"""

    def supported_methods(self) -> set[str]:
        return {"validate_images", "create_image_metadata"}

    def execute(self, df: pl.DataFrame, method: str, **kwargs: object) -> pl.DataFrame:
        """メソッド名に応じて処理を実行する。

        Args:
            df: 入力 DataFrame
            method: メソッド名
            **kwargs: メソッド固有のパラメータ

        Returns:
            処理結果の DataFrame

        時間計算量: O(N)
        空間計算量: O(N)
        """
        column = str(kwargs.get("column", "image_path"))
        if method == "validate_images":
            return self._validate_images(df, column)
        elif method == "create_image_metadata":
            return self._create_image_metadata(df, column)
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def _validate_images(df: pl.DataFrame, column: str) -> pl.DataFrame:
        """画像パスの存在を確認し、__image_valid__ カラムを追加する。

        時間計算量: O(N)
        空間計算量: O(N)
        """
        paths = df[column].to_list()
        valid = [Path(p).exists() for p in paths]
        return df.with_columns(pl.Series("__image_valid__", valid))

    @staticmethod
    def _create_image_metadata(df: pl.DataFrame, column: str) -> pl.DataFrame:
        """画像の幅・高さ・チャネル数を取得して追加する。

        時間計算量: O(N) — 各画像を開いてサイズを取得
        空間計算量: O(N)
        """
        paths = df[column].to_list()
        widths: list[int] = []
        heights: list[int] = []
        channels: list[int] = []

        for p in paths:
            try:
                with Image.open(p) as img:
                    w, h = img.size
                    mode_channels = {"RGB": 3, "RGBA": 4, "L": 1}
                    c = mode_channels.get(img.mode, 3)
                    widths.append(w)
                    heights.append(h)
                    channels.append(c)
            except Exception:
                widths.append(0)
                heights.append(0)
                channels.append(0)

        return df.with_columns(
            pl.Series("__image_width__", widths),
            pl.Series("__image_height__", heights),
            pl.Series("__image_channels__", channels),
        )
