"""
PipelineVisualizer — DAG を Mermaid JS inline HTML で可視化する。

実行のたびに pipeline_dag.html を生成する。
ブラウザで開くだけで DAG がレンダリングされる（追加ツール不要）。
"""

from pathlib import Path

from src.domain.data.preprocessor import Node

# Mermaid JS の CDN URL（オフライン環境の場合は inline embed に変更が必要）
_MERMAID_SCRIPT = "https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>Pipeline DAG</title>
<script src="{mermaid_script}"></script>
<script>mermaid.initialize({{startOnLoad: true}});</script>
</head>
<body>
<h1>Pipeline DAG</h1>
<div class="mermaid">
{mermaid_src}
</div>
</body>
</html>
"""


class PipelineVisualizer:
    """Node リストから Mermaid DAG HTML を生成する。"""

    def __init__(self, nodes: list[Node]) -> None:
        self._nodes = nodes

    def to_mermaid(self) -> str:
        """Mermaid graph LR ソースを生成する。"""
        lines = ["graph LR"]

        for node in self._nodes:
            if node.is_input:
                label = f"{node.id}([{node.id}<br/>input])"
            elif "output" in node.resolver_cfg:
                label = f"{node.id}[[{node.id}<br/>OUTPUT]]"
            else:
                resolver = next(iter(node.resolver_cfg), "?")
                method = ""
                if isinstance(node.resolver_cfg.get(resolver), dict):
                    method = str(node.resolver_cfg[resolver].get("method", ""))
                label = f"{node.id}[{node.id}<br/>{resolver}:{method}]"
            lines.append(f"  {node.id}{label}")

        for node in self._nodes:
            for dep in node.from_nodes:
                lines.append(f"  {dep} --> {node.id}")

        return "\n".join(lines)

    def save_html(self, path: Path) -> None:
        """pipeline_dag.html を生成して保存する。"""
        mermaid_src = self.to_mermaid()
        html = _HTML_TEMPLATE.format(
            mermaid_script=_MERMAID_SCRIPT,
            mermaid_src=mermaid_src,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
