"""
Phase 5: GradCAMUseCase のテスト。

なぜこのテストが必要か:
  - GradCAMUseCase は GradCAMAnalyzer Protocol に依存し、
    model_dir + 画像指定 → GradCAM 生成 + metainfo.yaml/README.md を担う。
  - UseCase 層は infrastructure に直接依存しないため、
    FakeAnalyzer を DI して Protocol 経由のテストを行う。
  - 出力ディレクトリに metainfo.yaml, README.md が生成されること。
  - per-directory .gitignore が生成されること。

時間計算量: O(N) — N: 分析対象画像数
空間計算量: O(N) — 結果リストのサイズ
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import yaml

from src.domain.model.gradcam import GradCAMResult
from src.usecase.analysis.gradcam import GradCAMUseCase


class FakeGradCAMAnalyzer:
    """テスト用の GradCAMAnalyzer 実装。"""

    def analyze(
        self,
        model_path: Path,
        image_paths: list[Path],
        output_dir: Path,
        target_layer: str | None = None,
    ) -> list[GradCAMResult]:
        results = []
        for img_path in image_paths:
            heatmap_path = output_dir / f"heatmap_{img_path.stem}.png"
            heatmap_path.parent.mkdir(parents=True, exist_ok=True)
            heatmap_path.write_bytes(b"fake_heatmap")
            results.append(
                GradCAMResult(
                    image_path=img_path,
                    heatmap_path=heatmap_path,
                    predicted_class=1,
                    confidence=0.85,
                )
            )
        return results


class TestGradCAMUseCase:
    def test_execute_generates_results_and_metainfo(self, tmp_path: Path) -> None:
        """execute が GradCAM 結果と metainfo.yaml を生成すること。"""
        model_path = tmp_path / "model.pt"
        model_path.touch()
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        for i in range(3):
            (image_dir / f"img_{i}.png").touch()
        output_dir = tmp_path / "gradcam_output"

        git_repo = MagicMock()
        git_repo.get_commit_hash.return_value = "a" * 40

        cfg = {
            "model_path": str(model_path),
            "image_dir": str(image_dir),
            "output_dir": str(output_dir),
            "target_layer": None,
            "job_id": "test_gradcam",
        }

        usecase = GradCAMUseCase(
            cfg=cfg,
            analyzer=FakeGradCAMAnalyzer(),
            git_repo=git_repo,
        )
        result = usecase.execute()

        assert len(result) == 3
        assert (output_dir / "metainfo.yaml").exists()
        assert (output_dir / "README.md").exists()
        assert (output_dir / ".gitignore").exists()

    def test_metainfo_contains_commit_hash(self, tmp_path: Path) -> None:
        """metainfo.yaml に commit_hash が含まれること。"""
        model_path = tmp_path / "model.pt"
        model_path.touch()
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        (image_dir / "img_0.png").touch()
        output_dir = tmp_path / "gradcam_output"

        git_repo = MagicMock()
        git_repo.get_commit_hash.return_value = "b" * 40

        cfg = {
            "model_path": str(model_path),
            "image_dir": str(image_dir),
            "output_dir": str(output_dir),
            "target_layer": None,
            "job_id": "test_gradcam",
        }

        GradCAMUseCase(
            cfg=cfg,
            analyzer=FakeGradCAMAnalyzer(),
            git_repo=git_repo,
        ).execute()

        metainfo = yaml.safe_load((output_dir / "metainfo.yaml").read_text())
        assert metainfo["commit_hash"] == "b" * 40
