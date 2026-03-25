"""
Phase 1: GradCAMResult dataclass / GradCAMAnalyzer Protocol のテスト。

なぜこのテストが必要か:
  - GradCAMResult は GradCAM 分析の出力を保持する dataclass。
    image_path, heatmap_path, predicted_class, confidence が正しく設定されることを確認する。
  - GradCAMAnalyzer は Protocol。runtime_checkable でインスタンスチェックが動くこと、
    analyze メソッドのシグネチャが正しいことを確認する。
  - Domain 層のため torch/torchvision に依存してはならない。
"""

from __future__ import annotations

from pathlib import Path

from src.domain.model.gradcam import GradCAMAnalyzer, GradCAMResult


class TestGradCAMResult:
    def test_create(self) -> None:
        """フィールドが正しくインスタンス化されること。"""
        result = GradCAMResult(
            image_path=Path("/data/image.png"),
            heatmap_path=Path("/output/heatmap.png"),
            predicted_class=1,
            confidence=0.95,
        )
        assert result.image_path == Path("/data/image.png")
        assert result.heatmap_path == Path("/output/heatmap.png")
        assert result.predicted_class == 1
        assert result.confidence == 0.95


class TestGradCAMAnalyzerProtocol:
    def test_protocol_is_runtime_checkable(self) -> None:
        """GradCAMAnalyzer が runtime_checkable であること。"""

        class FakeAnalyzer:
            def analyze(
                self,
                model_path: Path,
                image_paths: list[Path],
                output_dir: Path,
                target_layer: str | None = None,
            ) -> list[GradCAMResult]:
                return []

        assert isinstance(FakeAnalyzer(), GradCAMAnalyzer)
