"""
PushNotebookUseCase — Notebook 生成 → kernel-metadata.json 生成 → Kaggle プッシュ。

処理フロー:
1. output_dir/{competition}/ を作成し、per-directory .gitignore を配置する
2. Jinja2 テンプレートから notebook.ipynb を生成する
3. kernel-metadata.json を生成する
4. Kaggle API の kernels_push() でアップロードする
5. README.md を生成する（ツリー構造 + メタ情報）

Kaggle API は KaggleApiPort Protocol で抽象化しており、
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
from src.usecase.kaggle_notebook.notebook_renderer import NotebookRenderer

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


class KaggleApiPort(Protocol):
    """Kaggle API の最小インターフェース。テスト時は MagicMock で差し替える。"""

    def kernels_push(self, folder: str) -> Any:
        """kernel-metadata.json が存在するディレクトリを Kaggle にプッシュする。"""
        ...


@dataclass(frozen=True)
class PushResult:
    """PushNotebookUseCase の実行結果。"""

    notebook_path: Path
    metadata_path: Path


class PushNotebookUseCase:
    """Notebook 生成 + Kaggle プッシュを担うユースケース。

    Args:
        cfg: Hydra DictConfig。以下のキーを使用する:
            - cfg.output_dir: 出力先ルートディレクトリ
            - cfg.notebook.competition: Kaggle competition slug
            - cfg.notebook.kernel_slug: Kaggle Kernel の URL slug
            - cfg.notebook.src_dataset: src/ を格納する Kaggle Dataset slug
            - cfg.notebook.enable_gpu: GPU 有効化フラグ
            - cfg.notebook.enable_internet: インターネット接続フラグ
            - cfg.kaggle_username: Kaggle ユーザー名
        kaggle_api: KaggleApiPort を実装したオブジェクト。
    """

    def __init__(self, cfg: DictConfig, kaggle_api: KaggleApiPort) -> None:
        self._cfg = cfg
        self._api = kaggle_api
        self._renderer = NotebookRenderer()

    def execute(self) -> PushResult:
        """Notebook を生成して Kaggle にプッシュする。

        Returns:
            PushResult（notebook_path, metadata_path）

        Raises:
            RuntimeError: Kaggle API の呼び出しが SystemExit を上げた場合。
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
        enable_internet: bool = bool(self._cfg.notebook.enable_internet)
        notebook_path = output_dir / "notebook.ipynb"
        self._renderer.render(
            output_path=notebook_path,
            competition=competition,
            src_dataset=src_dataset,
            enable_internet=enable_internet,
        )
        logger.info("Notebook generated: %s", notebook_path)
        return notebook_path

    def _generate_kernel_metadata(self, output_dir: Path, competition: str) -> Path:
        """kernel-metadata.json を生成する。

        Kaggle の kernels push コマンドが要求するフォーマットに従う。
        参照: https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md
        """
        username: str = str(self._cfg.get("kaggle_username", ""))
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
            "dataset_sources": [f"{username}/{src_dataset}" if username else src_dataset],
            "competition_sources": [competition],
            "kernel_sources": [],
        }

        metadata_path = output_dir / "kernel-metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        logger.info("kernel-metadata.json generated: %s", metadata_path)
        return metadata_path

    def _push(self, output_dir: Path) -> None:
        """Kaggle API の kernels_push() でアップロードする。"""
        try:
            self._api.kernels_push(str(output_dir))
            logger.info("Pushed to Kaggle: %s", output_dir)
        except SystemExit as e:
            raise RuntimeError(
                "Kaggle API の呼び出しに失敗しました。認証情報を確認してください。"
            ) from e

    def _write_readme(self, output_dir: Path) -> None:
        """README.md にツリー構造を出力する。"""
        lines: list[str] = [
            f"# Kaggle Notebook Push — `{output_dir.name}`",
            "",
            "## Output Files",
            "",
            "```",
            output_dir.name + "/",
        ]
        lines += build_tree_lines(output_dir)
        lines.append("```")
        (output_dir / "README.md").write_text("\n".join(lines) + "\n")
