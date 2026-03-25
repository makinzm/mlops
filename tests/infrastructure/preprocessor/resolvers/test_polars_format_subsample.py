"""
PolarsResolver の format_column / stratified_subsample のテスト。

なぜこのテストが必要か:
  - format_column: テンプレート文字列でカラム値を埋め込んだ新カラムを作成する。
    画像パスの構築など、文字列結合が必要な前処理で汎用的に使う。
  - stratified_subsample: 層化サブサンプリングでデータを削減する。
    大規模データセットでの実験高速化に使う。ラベル分布を維持する必要がある。

時間計算量: O(N)
空間計算量: O(N)
"""

from __future__ import annotations

import polars as pl

from src.infrastructure.preprocessor.resolvers.polars_resolver import PolarsResolver


class TestFormatColumn:
    def test_format_column_creates_new_column(self) -> None:
        """テンプレートでカラム値を埋め込んだ新カラムが作成されること。"""
        df = pl.DataFrame({"id": ["abc", "def", "ghi"]})
        resolver = PolarsResolver()
        result = resolver.execute(
            df,
            "format_column",
            template="data/train/{id}.tif",
            source_col="id",
            output_col="image_path",
        )
        assert "image_path" in result.columns
        assert result["image_path"].to_list() == [
            "data/train/abc.tif",
            "data/train/def.tif",
            "data/train/ghi.tif",
        ]

    def test_format_column_preserves_existing_columns(self) -> None:
        """既存カラムが保持されること。"""
        df = pl.DataFrame({"id": ["a"], "label": [1]})
        resolver = PolarsResolver()
        result = resolver.execute(
            df,
            "format_column",
            template="images/{id}.png",
            source_col="id",
            output_col="path",
        )
        assert "id" in result.columns
        assert "label" in result.columns
        assert "path" in result.columns


class TestStratifiedSubsample:
    def test_reduces_row_count(self) -> None:
        """指定 fraction でデータが削減されること。"""
        df = pl.DataFrame(
            {
                "id": list(range(100)),
                "label": [0] * 50 + [1] * 50,
            }
        )
        resolver = PolarsResolver()
        result = resolver.execute(
            df,
            "stratified_subsample",
            fraction=0.1,
            stratify_col="label",
            seed=42,
        )
        assert len(result) < len(df)
        assert len(result) == 10  # 100 * 0.1

    def test_maintains_label_distribution(self) -> None:
        """ラベル分布が維持されること。"""
        df = pl.DataFrame(
            {
                "id": list(range(1000)),
                "label": [0] * 700 + [1] * 300,
            }
        )
        resolver = PolarsResolver()
        result = resolver.execute(
            df,
            "stratified_subsample",
            fraction=0.1,
            stratify_col="label",
            seed=42,
        )
        label_counts = result["label"].value_counts().sort("label")
        count_0 = label_counts.filter(pl.col("label") == 0)["count"][0]
        count_1 = label_counts.filter(pl.col("label") == 1)["count"][0]
        # 70:30 の比率が大体維持されること（±10%）
        ratio = count_0 / max(count_1, 1)
        assert 1.5 < ratio < 3.5  # 70/30 ≈ 2.33

    def test_seed_reproducibility(self) -> None:
        """同じ seed で同じ結果が得られること。"""
        df = pl.DataFrame(
            {
                "id": list(range(100)),
                "label": [0] * 50 + [1] * 50,
            }
        )
        resolver = PolarsResolver()
        r1 = resolver.execute(
            df, "stratified_subsample", fraction=0.2, stratify_col="label", seed=42
        )
        r2 = resolver.execute(
            df, "stratified_subsample", fraction=0.2, stratify_col="label", seed=42
        )
        assert r1["id"].to_list() == r2["id"].to_list()

    def test_fraction_1_returns_all(self) -> None:
        """fraction=1.0 で全データが返されること。"""
        df = pl.DataFrame({"id": [1, 2, 3], "label": [0, 1, 0]})
        resolver = PolarsResolver()
        result = resolver.execute(
            df, "stratified_subsample", fraction=1.0, stratify_col="label", seed=42
        )
        assert len(result) == 3
