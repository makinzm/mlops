"""
GradCAMUseCase — GradCAM 分析のユースケース。

GradCAMAnalyzer Protocol と GitRepository Protocol を受け取り、
model_path + 画像指定 → GradCAM ヒートマップ生成 + metainfo.yaml / README.md 保存を担う。

出力ディレクトリ構造:
  {output_dir}/
    ├── .gitignore
    ├── metainfo.yaml
    ├── README.md
    └── gradcam_*.png

時間計算量: O(N * P) — N: 分析対象画像数, P: モデルのフォワード+バックワードパス
空間計算量: O(N * H * W) — ヒートマップ保存
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.domain.model.gradcam import GradCAMAnalyzer, GradCAMResult
from src.domain.repository.git import GitRepository

logger = logging.getLogger(__name__)

_GRADCAM_DIR_GITIGNORE = """\
*
!.gitignore
!*.yaml
!*.md
!*.png
!*/
"""


class GradCAMUseCase:
    """GradCAM 分析を実行し、結果と metainfo を保存する。"""

    def __init__(
        self,
        cfg: dict[str, Any],
        analyzer: GradCAMAnalyzer,
        git_repo: GitRepository,
    ) -> None:
        self._cfg = cfg
        self._analyzer = analyzer
        self._git_repo = git_repo

    def execute(self) -> list[GradCAMResult]:
        """GradCAM 分析を実行する。

        時間計算量: O(N * P)
        空間計算量: O(N * H * W)

        Returns:
            GradCAMResult のリスト
        """
        cfg = self._cfg
        model_path = Path(cfg["model_path"])
        image_dir = Path(cfg["image_dir"])
        output_dir = Path(cfg["output_dir"])
        target_layer: str | None = cfg.get("target_layer")
        job_id: str = cfg.get("job_id", "gradcam")

        output_dir.mkdir(parents=True, exist_ok=True)

        # .gitignore 配置
        gitignore_path = output_dir / ".gitignore"
        gitignore_path.write_text(_GRADCAM_DIR_GITIGNORE)

        # 画像パスリスト
        image_paths = sorted(
            p
            for p in image_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        )

        commit_hash = self._git_repo.get_commit_hash()
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

        # GradCAM 実行
        results = self._analyzer.analyze(
            model_path=model_path,
            image_paths=image_paths,
            output_dir=output_dir,
            target_layer=target_layer,
        )

        # metainfo.yaml
        metainfo = {
            "job_id": job_id,
            "timestamp": timestamp,
            "commit_hash": commit_hash,
            "model_path": str(model_path),
            "num_images": len(results),
            "target_layer": target_layer,
            "results": [
                {
                    "image": str(r.image_path),
                    "heatmap": str(r.heatmap_path),
                    "predicted_class": r.predicted_class,
                    "confidence": r.confidence,
                }
                for r in results
            ],
        }
        (output_dir / "metainfo.yaml").write_text(
            yaml.dump(metainfo, allow_unicode=True, sort_keys=False)
        )

        # README.md
        lines = [
            f"# GradCAM Analysis — `{job_id}`",
            "",
            f"- commit: `{commit_hash}`",
            f"- model: `{model_path}`",
            f"- images: {len(results)}",
            "",
            "## Results",
            "",
            "| Image | Predicted | Confidence |",
            "|-------|-----------|------------|",
        ]
        for r in results:
            lines.append(f"| {r.image_path.name} | {r.predicted_class} | {r.confidence:.4f} |")
        (output_dir / "README.md").write_text("\n".join(lines) + "\n")

        logger.info(f"GradCAM 分析完了: {len(results)} 画像, output={output_dir}")
        return results
