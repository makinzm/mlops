"""
学習前入力バリデーション。

学習を開始する前にデータ・次元・パスを検証し、
問題がある場合は明確なエラーメッセージ（修正方法を含む）を返す。

チェック項目:
  1. fold ディレクトリの存在
  2. parquet ファイルの存在と必要カラム
  3. 画像パスの存在（サンプルチェック）
  4. ラベルの範囲チェック
  5. 画像サイズとモデル期待サイズの比較
  6. backbone 出力次元の検証

時間計算量: O(N) — N: サンプル数（画像パスチェックはサンプリング）
空間計算量: O(1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from PIL import Image

from src.domain.model.backbone import BackboneConfig
from src.infrastructure.trainer.backbone_registry import check_dimensions

logger = logging.getLogger(__name__)

_MAX_IMAGE_CHECK_SAMPLES = 10


@dataclass
class ValidationIssue:
    """バリデーション結果の 1 件。

    Attributes:
        field: 問題のあるフィールド名
        message: エラーメッセージ（修正方法を含む）
        severity: "error" or "warning"
    """

    field: str
    message: str
    severity: str  # "error" | "warning"


def validate_training_inputs(
    preprocess_output_dir: Path,
    backbone_config: BackboneConfig,
    target_col: str,
    image_path_col: str,
    num_classes: int,
) -> list[ValidationIssue]:
    """学習前にデータを検証し、問題のリストを返す。

    Args:
        preprocess_output_dir: fold_N/ を含む前処理出力ディレクトリ
        backbone_config: backbone 設定
        target_col: ターゲットカラム名
        image_path_col: 画像パスカラム名
        num_classes: 分類クラス数

    Returns:
        ValidationIssue のリスト（空なら問題なし）

    時間計算量: O(F * S) — F: fold 数, S: サンプル数
    空間計算量: O(1)
    """
    issues: list[ValidationIssue] = []

    # 1. fold ディレクトリの存在チェック
    fold_dirs = sorted(preprocess_output_dir.glob("fold_*"))
    if not fold_dirs:
        issues.append(
            ValidationIssue(
                field="preprocess_output_dir",
                message=(
                    f"fold ディレクトリが見つかりません: {preprocess_output_dir}\n"
                    f"'fold_N/' という名前のサブディレクトリが必要です。\n"
                    f"修正: preprocess_output_dir の設定を確認してください。"
                ),
                severity="error",
            )
        )
        return issues  # これ以上チェック不可

    for fold_dir in fold_dirs:
        # 2. parquet ファイルの存在チェック
        for parquet_name in ("train.parquet", "test.parquet"):
            parquet_path = fold_dir / parquet_name
            if not parquet_path.exists():
                issues.append(
                    ValidationIssue(
                        field=f"{fold_dir.name}/{parquet_name}",
                        message=f"{parquet_path} が見つかりません。前処理を実行してください。",
                        severity="error",
                    )
                )
                continue

            df = pl.read_parquet(parquet_path)

            # 3. 必要カラムの存在チェック
            for col in (image_path_col, target_col):
                if col not in df.columns:
                    issues.append(
                        ValidationIssue(
                            field=col,
                            message=(
                                f"カラム '{col}' が {parquet_path} に存在しません。\n"
                                f"利用可能なカラム: {df.columns}\n"
                                f"修正: target_col または image_path_col の設定を確認してください。"
                            ),
                            severity="error",
                        )
                    )
                    continue

            if image_path_col not in df.columns or target_col not in df.columns:
                continue

            # 4. 画像パスの存在チェック（サンプル）
            image_paths = df[image_path_col].to_list()
            sample_paths = image_paths[:_MAX_IMAGE_CHECK_SAMPLES]
            missing = [p for p in sample_paths if not Path(p).exists()]
            if missing:
                issues.append(
                    ValidationIssue(
                        field=image_path_col,
                        message=(
                            f"{fold_dir.name}/{parquet_name}: "
                            f"{len(missing)}/{len(sample_paths)} 枚の画像が見つかりません。\n"
                            f"例: {missing[0]}\n"
                            f"修正: 画像ファイルのパスが正しいか確認してください。"
                        ),
                        severity="error",
                    )
                )

            # 5. ラベルの範囲チェック
            labels = df[target_col].to_list()
            invalid_labels = [v for v in labels if v < 0 or v >= num_classes]
            if invalid_labels:
                issues.append(
                    ValidationIssue(
                        field=target_col,
                        message=(
                            f"{fold_dir.name}/{parquet_name}: "
                            f"ラベルが [0, {num_classes}) の範囲外です。\n"
                            f"検出された不正なラベル: {sorted(set(invalid_labels))[:5]}\n"
                            f"修正: num_classes={num_classes} を正しい値に変更するか、"
                            f"ラベルの前処理を確認してください。"
                        ),
                        severity="error",
                    )
                )

            # 6. 画像サイズチェック（最初の有効な画像のみ）
            if parquet_name == "train.parquet":
                valid_paths = [p for p in sample_paths if Path(p).exists()]
                if valid_paths:
                    try:
                        with Image.open(valid_paths[0]) as img:
                            actual_w, actual_h = img.size
                        expected = backbone_config.image_size
                        ratio = max(actual_w, actual_h) / expected
                        if ratio > 4.0 or ratio < 0.25:
                            issues.append(
                                ValidationIssue(
                                    field="backbone.image_size",
                                    message=(
                                        f"画像サイズ ({actual_w}x{actual_h}) と "
                                        f"モデル期待サイズ ({expected}x{expected}) が "
                                        f"大きく異なります（{ratio:.1f}倍）。\n"
                                        f"リサイズは自動で行われますが、品質が劣化する可能性があります。\n"
                                        f"修正: backbone.image_size を画像サイズに近い値"
                                        f"（例: {min(actual_w, actual_h)}）に変更してください。"
                                    ),
                                    severity="warning",
                                )
                            )
                    except Exception:
                        pass

    # 7. backbone 次元検証
    dim_info = check_dimensions(backbone_config)
    if not dim_info.is_valid:
        issues.append(
            ValidationIssue(
                field="backbone",
                message=(
                    f"backbone 出力次元ミスマッチ: {dim_info.message}\n"
                    f"backbone '{dim_info.backbone_name}' は "
                    f"{dim_info.output_features} 次元を出力しますが、"
                    f"classifier は "
                    f"{dim_info.expected_features} 次元を期待しています。"
                ),
                severity="error",
            )
        )

    return issues
