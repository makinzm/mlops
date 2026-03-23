"""
ステージングディレクトリ管理とファイルコピーのユーティリティ。

CreateSourceDatasetUseCase / UpdateSourceDatasetUseCase の共通ロジックを切り出す。

ステージング戦略:
  - .staging/source_dataset_{YYYYMMDD_HHMMSS}/ を作成する
  - src/ と conf/ を .kaggleignore でフィルタしながらコピーする
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

    マッチング戦略:
    - 末尾 / があるパターン（ディレクトリ指定）: パスの各コンポーネントと照合する
      例: `__pycache__/` → parts に `__pycache__` があれば一致
    - 末尾 / がないパターン: ファイル名・ディレクトリ名の各コンポーネントと照合する
      例: `*.pyc` → rel_path.name が一致すれば除外

    Args:
        rel_path: src_dir からの相対パス。
        patterns: .kaggleignore から読み込んだパターンリスト（コメント・空行なし）。
    """
    parts = rel_path.parts
    for pattern in patterns:
        # 末尾 / を除いた実効パターン
        effective = pattern.rstrip("/")
        # パスの各コンポーネント（ディレクトリ名・ファイル名）と照合
        for part in parts:
            if fnmatch.fnmatch(part, effective):
                return True
    return False


def _copy_dir_to_staging(
    source_dir: Path,
    staging_dir: Path,
    patterns: list[str],
) -> None:
    """source_dir の内容を staging_dir/{source_dir.name}/ にコピーする（内部ヘルパー）。

    .kaggleignore パターンにマッチするファイル・ディレクトリはコピーしない。

    Args:
        source_dir: コピー元ディレクトリ。
        staging_dir: コピー先のステージングディレクトリ。
        patterns: .kaggleignore から読み込んだ除外パターンリスト。
    """
    dest = staging_dir / source_dir.name
    dest.mkdir(parents=True, exist_ok=True)

    for src_file in source_dir.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(source_dir)
        if _is_ignored(rel, patterns):
            logger.debug("Ignored (kaggleignore): %s", rel)
            continue
        dest_file = dest / rel
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest_file)

    logger.info("Copied %s -> %s", source_dir, dest)


def copy_to_staging(
    src_dir: Path,
    conf_dir: Path,
    staging_dir: Path,
    patterns: list[str],
    requirements_path: Path | None = None,
    extra_dirs: list[Path] | None = None,
) -> None:
    """src_dir と conf_dir の内容を staging_dir にコピーする。

    .kaggleignore パターンにマッチするファイル・ディレクトリはコピーしない。
    requirements_path が存在する場合は staging_dir/requirements.txt にもコピーする。
    extra_dirs が指定された場合はそれらのディレクトリもコピーする。

    ステージング後の構造:
      staging_dir/
        src/          ← src_dir の中身
        conf/         ← conf_dir の中身
        models/...    ← extra_dirs の中身（指定された場合）
        requirements.txt  ← requirements_path が存在する場合

    Args:
        src_dir: コピー元 src/ ディレクトリ。
        conf_dir: コピー元 conf/ ディレクトリ。
        staging_dir: コピー先のステージングディレクトリ。
        patterns: .kaggleignore から読み込んだ除外パターンリスト。
        requirements_path: requirements.txt のパス（省略時はコピーしない）。
        extra_dirs: 追加でコピーするディレクトリのリスト（省略時はなし）。
    """
    _copy_dir_to_staging(src_dir, staging_dir, patterns)
    _copy_dir_to_staging(conf_dir, staging_dir, patterns)

    for extra_dir in extra_dirs or []:
        if extra_dir.exists():
            _copy_dir_to_staging(extra_dir, staging_dir, patterns)

    if requirements_path is not None and requirements_path.exists():
        shutil.copy2(requirements_path, staging_dir / "requirements.txt")
        logger.info("Copied %s -> %s/requirements.txt", requirements_path, staging_dir)


def cleanup_staging_dir(staging_dir: Path) -> None:
    """ステージングサブディレクトリを削除する（成功時のみ呼ぶ）。

    Args:
        staging_dir: 削除するステージングサブディレクトリ。
    """
    shutil.rmtree(staging_dir, ignore_errors=True)
    logger.info("Staging dir removed: %s", staging_dir)
