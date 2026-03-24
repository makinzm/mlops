"""
PushNotebookUseCase — Notebook 生成 → kernel-metadata.json 生成 → プラットフォームへプッシュ。

処理フロー:
1. output_dir/{competition}/ を作成し、per-directory .gitignore を配置する
2. Jinja2 テンプレートから notebook.ipynb を生成する
3. kernel-metadata.json を生成する
4. NotebookPlatformPort の kernels_push() でアップロードする
5. README.md を生成する（ツリー構造 + メタ情報）

Notebook API は NotebookPlatformPort Protocol で抽象化しており、
テスト時は MagicMock に差し替えて CI で実際の通信が発生しないようにする。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from omegaconf import DictConfig

from src.usecase._utils import build_tree_lines
from src.usecase.notebook.notebook_renderer import NotebookRenderer

logger = logging.getLogger(__name__)

_PUSH_NOTEBOOK_GITIGNORE = """\
*
!.gitignore
!*.yaml
!*.md
!*.ipynb
!*.json
!*/
"""


class NotebookPlatformPort(Protocol):
    """Notebook プラットフォーム API の最小インターフェース。テスト時は MagicMock で差し替える。"""

    def kernels_push(self, folder: str) -> Any:
        """kernel-metadata.json が存在するディレクトリをプラットフォームにプッシュする。"""
        ...


@dataclass(frozen=True)
class PushResult:
    """PushNotebookUseCase の実行結果。"""

    notebook_path: Path
    metadata_path: Path


class PushNotebookUseCase:
    """Notebook 生成 + プラットフォームプッシュを担うユースケース。

    Args:
        cfg: Hydra DictConfig。以下のキーを使用する:
            - cfg.output_dir: 出力先ルートディレクトリ
            - cfg.notebook.competition: competition slug
            - cfg.notebook.kernel_slug: Kernel の URL slug
            - cfg.notebook.src_dataset: src/ を格納する Dataset slug
            - cfg.notebook.enable_gpu: GPU 有効化フラグ
            - cfg.notebook.enable_internet: インターネット接続フラグ
            - cfg.platform_username: ユーザー名
        platform_api: NotebookPlatformPort を実装したオブジェクト。
    """

    def __init__(self, cfg: DictConfig, platform_api: NotebookPlatformPort) -> None:
        self._cfg = cfg
        self._api = platform_api
        self._renderer = NotebookRenderer()

    def execute(self) -> PushResult:
        """Notebook を生成してプラットフォームにプッシュする。

        Returns:
            PushResult（notebook_path, metadata_path）

        Raises:
            RuntimeError: API の呼び出しが SystemExit を上げた場合。
        """
        competition: str = str(self._cfg.notebook.competition)
        output_dir = Path(str(self._cfg.output_dir)) / competition
        output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_gitignore(output_dir)
        notebook_path = self._render_notebook(output_dir, competition)
        metadata_path = self._generate_kernel_metadata(output_dir, competition)
        self._push(output_dir)
        self._write_readme(output_dir)

        return PushResult(notebook_path=notebook_path, metadata_path=metadata_path)

    def _setup_gitignore(self, output_dir: Path) -> None:
        """per-directory .gitignore を配置する。既存ファイルは上書きしない。"""
        gitignore_path = output_dir / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text(_PUSH_NOTEBOOK_GITIGNORE)

    def _render_notebook(self, output_dir: Path, competition: str) -> Path:
        """Jinja2 テンプレートから notebook.ipynb を生成する。"""
        src_dataset: str = str(self._cfg.notebook.src_dataset)
        platform_username: str = str(self._cfg.get("platform_username", ""))
        enable_internet: bool = bool(self._cfg.notebook.enable_internet)
        notebook_path = output_dir / "notebook.ipynb"
        recipe: str = str(self._cfg.notebook.get("recipe", "all_after_download"))
        extra_datasets_raw = self._cfg.notebook.get("extra_datasets") or []
        extra_datasets: list[dict[str, str]] = [
            {"slug": str(d.slug), "mount_path": str(d.mount_path)} for d in extra_datasets_raw
        ]
        self._renderer.render(
            output_path=notebook_path,
            competition=competition,
            src_dataset=src_dataset,
            kaggle_username=platform_username,
            enable_internet=enable_internet,
            recipe=recipe,
            extra_datasets=extra_datasets,
        )
        logger.info("Notebook generated: %s", notebook_path)
        return notebook_path

    def _generate_kernel_metadata(self, output_dir: Path, competition: str) -> Path:
        """kernel-metadata.json を生成する。

        Notebook プラットフォームの kernels push コマンドが要求するフォーマットに従う。
        """
        username: str = str(self._cfg.get("platform_username", ""))
        kernel_slug: str = str(self._cfg.notebook.kernel_slug)
        src_dataset: str = str(self._cfg.notebook.src_dataset)
        enable_gpu: bool = bool(self._cfg.notebook.enable_gpu)
        enable_internet: bool = bool(self._cfg.notebook.enable_internet)

        metadata: dict[str, Any] = {
            "id": f"{username}/{kernel_slug}" if username else kernel_slug,
            "title": kernel_slug,
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": enable_gpu,
            "enable_internet": enable_internet,
            "dataset_sources": [f"{username}/{src_dataset}" if username else src_dataset]
            + [
                f"{username}/{d.slug}" if username else str(d.slug)
                for d in (self._cfg.notebook.get("extra_datasets") or [])
            ],
            "competition_sources": [competition],
            "kernel_sources": [],
        }

        metadata_path = output_dir / "kernel-metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        logger.info("kernel-metadata.json generated: %s", metadata_path)
        return metadata_path

    def _push(self, output_dir: Path) -> None:
        """Notebook プラットフォームの kernels_push() でアップロードする。"""
        try:
            self._api.kernels_push(str(output_dir))
            logger.info("Pushed notebook: %s", output_dir)
        except SystemExit as e:
            raise RuntimeError(
                "Notebook API の呼び出しに失敗しました。認証情報を確認してください。"
            ) from e

    def _write_readme(self, output_dir: Path) -> None:
        """README.md にツリー構造を出力する。"""
        lines: list[str] = [
            f"# Notebook Push — `{output_dir.name}`",
            "",
            "## Output Files",
            "",
            "```",
            output_dir.name + "/",
        ]
        lines += build_tree_lines(output_dir)
        lines.append("```")
        (output_dir / "README.md").write_text("\n".join(lines) + "\n")
