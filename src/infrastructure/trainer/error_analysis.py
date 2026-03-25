"""
共通 Error Analysis — LightGBM / Vision で共有する TP/TN/FP/FN サンプリング。

torch に依存しない（numpy/pandas/polars のみ）ため、
tabular モデルと vision モデルの両方から使える。

時間計算量: O(N log N) — ソート
空間計算量: O(N)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


def write_error_analysis(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_samples: int,
    output_path: Path,
    extra_columns: dict[str, Any] | None = None,
    threshold: float = 0.5,
) -> None:
    """4 種サンプリング（TP/TN/FP/FN）を parquet に保存する。

    Args:
        predictions: shape=(N,) の予測確率
        labels: shape=(N,) の正解ラベル（0 or 1）
        n_samples: 各カテゴリからサンプリングする件数
        output_path: 出力 parquet パス
        extra_columns: 追加カラム（feature 値や画像パスなど）
        threshold: 二値分類の閾値

    時間計算量: O(N log N)
    空間計算量: O(N)
    """
    pred_label = (predictions >= threshold).astype(int)
    y = np.asarray(labels)

    result = pd.DataFrame(
        {
            "target": y,
            "predicted_proba": predictions,
            "predicted_label": pred_label,
            "is_correct": (pred_label == y).astype(int),
            "error_magnitude": np.abs(predictions - y),
        }
    )

    if extra_columns:
        for col_name, col_values in extra_columns.items():
            result[col_name] = col_values

    def _label(t: int, p: int) -> str:
        if t == 1 and p == 1:
            return "TP"
        if t == 0 and p == 0:
            return "TN"
        if t == 0 and p == 1:
            return "FP"
        return "FN"

    result["sample_type"] = [_label(int(t), int(p)) for t, p in zip(y, pred_label)]

    samples: list[pd.DataFrame] = []
    for stype in ("TP", "TN", "FP", "FN"):
        subset = result[result["sample_type"] == stype]
        if stype in ("FP", "FN"):
            subset = subset.sort_values("error_magnitude", ascending=False)
        else:
            subset = subset.sort_values("error_magnitude", ascending=True)
        samples.append(subset.head(n_samples))

    combined = pd.concat(samples, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(combined).write_parquet(output_path)
