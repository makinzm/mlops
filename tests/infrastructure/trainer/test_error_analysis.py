"""
共通 error_analysis のテスト。

なぜこのテストが必要か:
  - write_error_analysis() が LightGBM と Vision で共通の error analysis を生成すること。
  - TP/TN/FP/FN の 4 種サンプリングが正しく動くこと。
  - 必要なカラムが全て含まれること。
  - LightGBM と Vision の両方から呼ばれる共通モジュールなので独立テストが必要。

時間計算量: O(N log N)
空間計算量: O(N)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.infrastructure.trainer.error_analysis import write_error_analysis


class TestWriteErrorAnalysis:
    def test_creates_parquet_file(self, tmp_path: Path) -> None:
        """parquet ファイルが作成されること。"""
        out_path = tmp_path / "error_analysis.parquet"
        preds = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        labels = np.array([1, 0, 0, 1, 1, 0])
        write_error_analysis(
            predictions=preds,
            labels=labels,
            n_samples=3,
            output_path=out_path,
            extra_columns={"image_path": [f"img_{i}.png" for i in range(6)]},
        )
        assert out_path.exists()

    def test_required_columns_present(self, tmp_path: Path) -> None:
        """必要なカラムが全て含まれること。"""
        out_path = tmp_path / "error_analysis.parquet"
        preds = np.array([0.9, 0.1, 0.8, 0.2])
        labels = np.array([1, 0, 0, 1])
        write_error_analysis(
            predictions=preds,
            labels=labels,
            n_samples=2,
            output_path=out_path,
        )
        df = pd.read_parquet(out_path)
        required = [
            "target",
            "predicted_proba",
            "predicted_label",
            "is_correct",
            "error_magnitude",
            "sample_type",
        ]
        for col in required:
            assert col in df.columns, f"カラム '{col}' が存在しません"

    def test_sample_types_are_valid(self, tmp_path: Path) -> None:
        """sample_type が TP/TN/FP/FN のいずれかであること。"""
        out_path = tmp_path / "error_analysis.parquet"
        preds = np.array([0.9, 0.1, 0.8, 0.2, 0.6, 0.4])
        labels = np.array([1, 0, 0, 1, 1, 0])
        write_error_analysis(
            predictions=preds,
            labels=labels,
            n_samples=3,
            output_path=out_path,
        )
        df = pd.read_parquet(out_path)
        assert set(df["sample_type"].unique()).issubset({"TP", "TN", "FP", "FN"})

    def test_extra_columns_preserved(self, tmp_path: Path) -> None:
        """extra_columns が保存されること。"""
        out_path = tmp_path / "error_analysis.parquet"
        preds = np.array([0.9, 0.1])
        labels = np.array([1, 0])
        write_error_analysis(
            predictions=preds,
            labels=labels,
            n_samples=2,
            output_path=out_path,
            extra_columns={"feature_a": [1.0, 2.0], "feature_b": [3.0, 4.0]},
        )
        df = pd.read_parquet(out_path)
        assert "feature_a" in df.columns
        assert "feature_b" in df.columns
