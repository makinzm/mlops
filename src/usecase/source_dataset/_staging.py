"""
ステージングディレクトリ管理とファイルコピーのユーティリティ。

CreateSourceDatasetUseCase / UpdateSourceDatasetUseCase の共通ロジックを切り出す。

ステージング戦略:
  - .staging/source_dataset_{YYYYMMDD_HHMMSS}/ を作成する
  - src/ を .kaggleignore でフィルタしながらコピーする
  - 成功時はステージングディレクトリを削除する（失敗時は残す）
  - /tmp は使わない（プロジェクトルート直下の .staging/ のみ使用）
"""

from __future__ import annotations

import fnmatch
import logging
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_STAGING_PREFIX = "source_dataset_"
_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def make_staging_dir(staging_root: Path) -> Path:
    """タイムスタンプ付きのステージングサブディレクトリを作成して返す。

    Args:
        staging_root: ステージングルートディレクトリ（例: .staging/）。

    Returns:
        作成したサブディレクトリのパス（例: .staging/source_dataset_20260316_120000/）。
    """
    timestamp = datetime.now().strftime(_TIMESTAMP_FORMAT)
    staging_dir = staging_root / f"{_STAGING_PREFIX}{timestamp}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Staging dir created: %s", staging_dir)
    return staging_dir


def load_kaggleignore_patterns(kaggleignore_path: Path | None) -> list[str]:
    """kaggleignore ファイルからパターンリストを読み込む。

    Args:
        kaggleignore_path: .kaggleignore ファイルのパス。None の場合は空リストを返す。

    Returns:
        除外パターンのリスト（コメント行・空行は除く）。
    """
    if kaggleignore_path is None or not kaggleignore_path.exists():
        return []
    lines = kaggleignore_path.read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def _is_ignored(rel_path: Path, patterns: list[str]) -> bool:
    """指定した相対パスが .kaggleignore パターンにマッチするか返す。

    ディレクトリ名・ファイル名・拡張子に対して fnmatch でマッチングを行う。
    `__pycache__/` のようなディレクトリパターンはパスの各コンポーネントと照合する。
    """
    parts = rel_path.parts
    for pattern in patterns:
        # ディレクトリパターン（末尾 / を除去して判定）
        dir_pattern = pattern.rstrip("/")
        for part in parts:
            if fnmatch.fnmatch(part, dir_pattern):
                return True
        # ファイル名全体へのマッチ
        if fnmatch.fnmatch(rel_path.name, pattern):
            return True
    return False


def copy_src_to_staging(
    src_dir: Path,
    staging_dir: Path,
    patterns: list[str],
) -> None:
    """src_dir の内容を staging_dir/{src_dir.name}/ にコピーする。

    .kaggleignore パターンにマッチするファイル・ディレクトリはコピーしない。

    Args:
        src_dir: コピー元ディレクトリ（例: src/）。
        staging_dir: コピー先のステージングディレクトリ。
        patterns: .kaggleignore から読み込んだ除外パターンリスト。
    """
    dest = staging_dir / src_dir.name
    dest.mkdir(parents=True, exist_ok=True)

    for src_file in src_dir.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src_dir)
        if _is_ignored(rel, patterns):
            logger.debug("Ignored (kaggleignore): %s", rel)
            continue
        dest_file = dest / rel
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest_file)

    logger.info("Copied %s -> %s", src_dir, dest)


def cleanup_staging_dir(staging_dir: Path) -> None:
    """ステージングサブディレクトリを削除する（成功時のみ呼ぶ）。

    Args:
        staging_dir: 削除するステージングサブディレクトリ。
    """
    shutil.rmtree(staging_dir, ignore_errors=True)
    logger.info("Staging dir removed: %s", staging_dir)
