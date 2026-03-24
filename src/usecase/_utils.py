"""usecase 層の共通ユーティリティ。"""

from pathlib import Path


def resolve_latest_dir(path_str: str) -> Path:
    """パス文字列中の 'latest' を最新タイムスタンプディレクトリに解決する。

    例）
      .../{job_id}_preprocess/latest/train_out
      → .../{job_id}_preprocess/20260315T180000/train_out

    'latest' が含まれない場合はそのまま Path に変換して返す。

    Raises:
        ValueError: 'latest' ディレクトリ配下にタイムスタンプディレクトリが存在しない場合
    """
    parts = Path(path_str).parts
    latest_indices = [i for i, p in enumerate(parts) if p == "latest"]
    if not latest_indices:
        return Path(path_str)

    idx = latest_indices[0]
    parent = Path(*parts[:idx])
    suffix = Path(*parts[idx + 1 :]) if len(parts) > idx + 1 else Path()

    candidates = sorted(parent.iterdir(), key=lambda p: p.name, reverse=True)
    dirs = [c for c in candidates if c.is_dir()]
    if not dirs:
        raise ValueError(f"No timestamp directory found under {parent}")

    resolved = dirs[0]
    return resolved / suffix if str(suffix) != "." else resolved


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
