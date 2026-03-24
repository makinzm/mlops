"""
GradCAM 関連のドメインデータクラスと Protocol。

GradCAMResult: GradCAM 分析結果（画像パス、ヒートマップパス、予測クラス、確信度）。
GradCAMAnalyzer: GradCAM 分析の Protocol。infrastructure 層で実装する。

設計方針:
  - Domain 層のため torch/torchvision/pytorch_grad_cam に依存しない。
  - GradCAMAnalyzer は runtime_checkable な Protocol。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class GradCAMResult:
    """1 画像分の GradCAM 分析結果。

    Attributes:
        image_path: 分析対象の元画像パス
        heatmap_path: 生成されたヒートマップ画像のパス
        predicted_class: モデルの予測クラス
        confidence: 予測の確信度（softmax 出力の最大値）
    """

    image_path: Path
    heatmap_path: Path
    predicted_class: int
    confidence: float


@runtime_checkable
class GradCAMAnalyzer(Protocol):
    """GradCAM 分析の抽象 Protocol。

    GradCAMAnalyzerImpl（infrastructure）がこの Protocol を満たす。
    UseCase 層はこの Protocol にのみ依存する。
    """

    def analyze(
        self,
        model_path: Path,
        image_paths: list[Path],
        output_dir: Path,
        target_layer: str | None = None,
    ) -> list[GradCAMResult]:
        """指定モデルで画像の GradCAM ヒートマップを生成する。

        Args:
            model_path: 学習済みモデルファイルのパス（.pt）
            image_paths: 分析対象の画像パスリスト
            output_dir: ヒートマップの出力先ディレクトリ
            target_layer: GradCAM のターゲットレイヤー名（None の場合自動検出）

        Returns:
            各画像の GradCAMResult リスト
        """
        ...
