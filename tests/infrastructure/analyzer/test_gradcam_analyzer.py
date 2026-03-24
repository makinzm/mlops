"""
Phase 5: GradCAMAnalyzerImpl のテスト。

なぜこのテストが必要か:
  - GradCAMAnalyzerImpl は pytorch_grad_cam ライブラリを使って
    GradCAM ヒートマップを生成する。
  - 合成画像 + 小さいモデルで実際にヒートマップが生成されることを確認する。
  - GradCAMAnalyzer Protocol を満たすことを確認する。
  - 出力の GradCAMResult が正しいフィールドを持つことを確認する。

時間計算量: O(N * P) — N: 画像数, P: モデルのフォワードパス計算量
空間計算量: O(P + H*W) — P: モデルパラメータ数, H*W: 画像解像度
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.domain.model.backbone import BackboneConfig
from src.domain.model.gradcam import GradCAMAnalyzer, GradCAMResult
from src.infrastructure.analyzer.gradcam_analyzer import GradCAMAnalyzerImpl
from src.infrastructure.trainer.backbone_registry import build_backbone, build_classifier


def _create_synthetic_images(image_dir: Path, num_images: int, seed: int = 42) -> list[Path]:
    """テスト用合成画像を生成して Path リストを返す。"""
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths: list[Path] = []
    for i in range(num_images):
        data = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(data)
        path = image_dir / f"img_{i:04d}.png"
        img.save(path)
        paths.append(path)
    return paths


def _make_vision_model(path: Path) -> None:
    """テスト用 Vision モデルを保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    config = BackboneConfig(backbone_name="simple_cnn", num_classes=2, pretrained=False)
    backbone, num_features = build_backbone(config)
    classifier = build_classifier(num_features, 2)
    model = nn.Sequential(backbone, nn.Flatten(), classifier)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "backbone_config": {
                "backbone_name": "simple_cnn",
                "num_classes": 2,
                "pretrained": False,
                "image_size": 32,
            },
        },
        path,
    )


class TestGradCAMAnalyzerImpl:
    def test_satisfies_protocol(self) -> None:
        """GradCAMAnalyzer Protocol を満たすこと。"""
        analyzer = GradCAMAnalyzerImpl()
        assert isinstance(analyzer, GradCAMAnalyzer)

    def test_analyze_returns_gradcam_results(self, tmp_path: Path) -> None:
        """analyze が GradCAMResult のリストを返すこと。"""
        model_path = tmp_path / "model.pt"
        _make_vision_model(model_path)
        image_paths = _create_synthetic_images(tmp_path / "images", 3)
        output_dir = tmp_path / "gradcam_output"

        analyzer = GradCAMAnalyzerImpl()
        results = analyzer.analyze(
            model_path=model_path,
            image_paths=image_paths,
            output_dir=output_dir,
        )
        assert len(results) == 3
        for result in results:
            assert isinstance(result, GradCAMResult)
            assert result.heatmap_path.exists()
            assert result.predicted_class >= 0
            assert 0.0 <= result.confidence <= 1.0
