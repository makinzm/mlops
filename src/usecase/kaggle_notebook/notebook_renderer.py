"""
NotebookRenderer — Jinja2 テンプレートから Jupyter Notebook (.ipynb) を生成する。

テンプレートは templates/notebook/pipeline.ipynb.j2 を使用する。
生成した ipynb は JSON として出力する。
"""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_TEMPLATES_DIR = Path(__file__).parent.parent.parent.parent / "templates" / "notebook"
_TEMPLATE_NAME = "pipeline.ipynb.j2"


class NotebookRenderer:
    """Jinja2 テンプレートから ipynb ファイルを生成する。"""

    def __init__(self, templates_dir: Path = _TEMPLATES_DIR) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            undefined=StrictUndefined,
            autoescape=False,  # ipynb は JSON のため HTML エスケープ不要
        )

    def render(
        self,
        output_path: Path,
        competition: str,
        src_dataset: str,
        kaggle_username: str,
        enable_internet: bool = True,
        recipe: str = "all_after_download",
    ) -> Path:
        """テンプレートを描画して ipynb ファイルを書き出す。

        Args:
            output_path: 出力先ファイルパス（例: outputs/push_notebook/titanic/notebook.ipynb）
            competition: Kaggle competition の slug（例: "titanic"）
            src_dataset: コードを格納する Kaggle Dataset の slug（例: "mlops-pipeline-src"）
            enable_internet: False のとき pip install をスキップする（オフラインコンペ用）
            recipe: パイプラインレシピ名（例: "inference_only"）

        Returns:
            書き出した ipynb ファイルのパス。
        """
        template = self._env.get_template(_TEMPLATE_NAME)
        rendered = template.render(
            competition=competition,
            src_dataset=src_dataset,
            kaggle_username=kaggle_username,
            enable_internet=enable_internet,
            recipe=recipe,
        )

        # JSON として parse してから再 dump することで整形済みファイルを生成する
        notebook = json.loads(rendered)
        output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
        return output_path
