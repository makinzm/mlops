"""usecase 層の共通ユーティリティ。"""

from pathlib import Path


def build_tree_lines(directory: Path, prefix: str = "") -> list[str]:
    """directory 配下のファイル・ディレクトリを ASCII ツリー形式で返す。"""
    entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
    lines: list[str] = []
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{entry.name}")
        if entry.is_dir():
            extension = "    " if is_last else "│   "
            lines += build_tree_lines(entry, prefix + extension)
    return lines
