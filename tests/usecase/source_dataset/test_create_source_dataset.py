"""
CreateSourceDatasetUseCase のテスト。

なぜこのテストが必要か:
  - src/ のコピー・フィルタリング・ステージング管理・リポジトリ呼び出しが
    正しく連携することを保証する。
  - 成功時にステージングディレクトリが削除されること、
    失敗時に残ることをテストで確認することで、デバッグ用の残存ロジックを保証する。
  - .kaggleignore によるフィルタリングが機能することを確認する。
  - SourceDatasetRepository は MagicMock で差し替えて、usecase 層が
    Kaggle に依存しないことを担保する。

fixture:
  - tmp_path: src_dir / staging_dir / kaggleignore をすべて tmp_path 以下に配置
  - MagicMock: SourceDatasetRepository の差し替え
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from src.usecase.source_dataset.create_source_dataset import CreateSourceDatasetUseCase


def _make_cfg(
    tmp_path: Path,
    kaggleignore_path: Path | None = None,
    conf_dir: Path | None = None,
) -> object:
    """CreateSourceDatasetUseCase 用の DictConfig を生成する。"""
    return OmegaConf.create(
        {
            "usecase": "create_source_dataset",
            "source_dataset": {
                "src_dir": str(tmp_path / "src"),
                "conf_dir": str(conf_dir) if conf_dir is not None else str(tmp_path / "conf"),
                "dataset_slug": "mlops-pipeline-src",
                "title": "mlops-pipeline-src",
                "license_name": "CC0-1.0",
                "ignorefile": str(kaggleignore_path) if kaggleignore_path else None,
            },
            "staging_dir": str(tmp_path / ".staging"),
            "platform_username": "testuser",
        }
    )


def _setup_src(tmp_path: Path) -> Path:
    """テスト用の src/ ディレクトリを作成する。"""
    src = tmp_path / "src"
    src.mkdir()
    (src / "__init__.py").write_text("")
    (src / "main.py").write_text("# main")
    sub = src / "domain"
    sub.mkdir()
    (sub / "__init__.py").write_text("")
    return src


class TestCreateSourceDatasetUseCaseExecute:
    """execute() メソッドの主要な振る舞いを検証する。"""

    def test_execute_copies_src_to_staging(self, tmp_path: Path) -> None:
        """execute() が create() を呼ぶ際に staging_dir が渡され src のファイルが含まれること。"""
        _setup_src(tmp_path)
        cfg = _make_cfg(tmp_path)
        captured: dict[str, Path] = {}

        mock_repo = MagicMock()

        def capture_create(staging_dir: Path, metadata: object) -> None:
            captured["staging_dir"] = staging_dir
            # コピー済みのファイルを確認してから終了（成功扱い）
            assert (staging_dir / "src" / "__init__.py").exists(), (
                "staging 内に src/__init__.py がコピーされていない"
            )

        mock_repo.create.side_effect = capture_create
        usecase = CreateSourceDatasetUseCase(cfg=cfg, repository=mock_repo)  # ty:ignore[invalid-argument-type]

        usecase.execute()

        assert "staging_dir" in captured, "create() が呼ばれていない"

    def test_execute_calls_repository_create(self, tmp_path: Path) -> None:
        """SourceDatasetRepository.create() が1回呼ばれること。"""
        _setup_src(tmp_path)
        cfg = _make_cfg(tmp_path)
        mock_repo = MagicMock()
        usecase = CreateSourceDatasetUseCase(cfg=cfg, repository=mock_repo)  # ty:ignore[invalid-argument-type]

        usecase.execute()

        mock_repo.create.assert_called_once()

    def test_execute_removes_staging_dir_on_success(self, tmp_path: Path) -> None:
        """成功時に staging subdir が削除されること。"""
        _setup_src(tmp_path)
        cfg = _make_cfg(tmp_path)
        mock_repo = MagicMock()
        usecase = CreateSourceDatasetUseCase(cfg=cfg, repository=mock_repo)  # ty:ignore[invalid-argument-type]

        usecase.execute()

        staging_root = tmp_path / ".staging"
        if staging_root.exists():
            remaining = list(staging_root.rglob("*"))
            assert len(remaining) == 0, f"成功後に staging に残骸がある: {remaining}"

    def test_execute_keeps_staging_dir_on_failure(self, tmp_path: Path) -> None:
        """失敗時に staging subdir が残ること（デバッグ用）。"""
        _setup_src(tmp_path)
        cfg = _make_cfg(tmp_path)
        mock_repo = MagicMock()
        mock_repo.create.side_effect = RuntimeError("upload failed")
        usecase = CreateSourceDatasetUseCase(cfg=cfg, repository=mock_repo)  # ty:ignore[invalid-argument-type]

        with pytest.raises(RuntimeError, match="upload failed"):
            usecase.execute()

        staging_root = tmp_path / ".staging"
        assert staging_root.exists(), "失敗時は staging が残るべき"
        subdirs = [d for d in staging_root.iterdir() if d.is_dir()]
        assert len(subdirs) >= 1, "失敗時は staging subdir が残るべき"

    def test_execute_respects_kaggleignore(self, tmp_path: Path) -> None:
        """.kaggleignore でフィルタされたファイルが staging にコピーされないこと。"""
        src = _setup_src(tmp_path)
        # __pycache__ ディレクトリを作成
        pycache = src / "__pycache__"
        pycache.mkdir()
        (pycache / "main.cpython-312.pyc").write_text("bytecode")

        # .kaggleignore を作成
        kaggleignore = tmp_path / ".kaggleignore"
        kaggleignore.write_text("__pycache__/\n*.pyc\n")

        cfg = _make_cfg(tmp_path, kaggleignore_path=kaggleignore)
        mock_repo = MagicMock()

        # staging dir の中身を確認するために create() 呼び出し時の staging_dir を記録
        captured: dict[str, Path] = {}

        def capture_create(staging_dir: Path, metadata: object) -> None:
            captured["staging_dir"] = staging_dir

        mock_repo.create.side_effect = capture_create
        usecase = CreateSourceDatasetUseCase(cfg=cfg, repository=mock_repo)  # ty:ignore[invalid-argument-type]

        usecase.execute()

        assert "staging_dir" in captured, "create() が呼ばれていない"
        staging_src = captured["staging_dir"] / "src"
        pycache_in_staging = staging_src / "__pycache__"
        assert not pycache_in_staging.exists(), "__pycache__/ が staging にコピーされている"

    def test_execute_uses_timestamp_subdir(self, tmp_path: Path) -> None:
        """staging subdir がタイムスタンプ形式（source_dataset_YYYYMMDD_HHMMSS）であること。"""
        _setup_src(tmp_path)
        cfg = _make_cfg(tmp_path)
        captured: dict[str, Path] = {}

        mock_repo = MagicMock()

        def capture_create(staging_dir: Path, metadata: object) -> None:
            captured["staging_dir"] = staging_dir

        mock_repo.create.side_effect = capture_create
        usecase = CreateSourceDatasetUseCase(cfg=cfg, repository=mock_repo)  # ty:ignore[invalid-argument-type]

        usecase.execute()

        assert "staging_dir" in captured
        subdir_name = captured["staging_dir"].name
        assert re.match(r"source_dataset_\d{8}_\d{6}$", subdir_name), (
            f"staging subdir 名が期待形式でない: {subdir_name}"
        )

    def test_execute_copies_conf_to_staging(self, tmp_path: Path) -> None:
        """conf/ も staging にコピーされること。

        なぜこのテストが必要か:
          - Kaggle Notebook 上では /kaggle/input/mlops-pipeline-src/conf/ から
            recipe yaml を読み込む。conf/ が staging に含まれないと Notebook 実行時に
            FileNotFoundError になる。
          - このテストで conf/ コピーが正しく機能することを保証する。
        """
        _setup_src(tmp_path)
        # conf/ を作成
        conf = tmp_path / "conf"
        conf.mkdir()
        recipe_dir = conf / "recipe"
        recipe_dir.mkdir()
        (recipe_dir / "base.yaml").write_text("steps: []")

        cfg = _make_cfg(tmp_path, conf_dir=conf)
        captured: dict[str, Path] = {}

        mock_repo = MagicMock()

        def capture_create(staging_dir: Path, metadata: object) -> None:
            captured["staging_dir"] = staging_dir
            # staging に conf/recipe/base.yaml が含まれていること
            assert (staging_dir / "conf" / "recipe" / "base.yaml").exists(), (
                "staging 内に conf/recipe/base.yaml がコピーされていない"
            )

        mock_repo.create.side_effect = capture_create
        usecase = CreateSourceDatasetUseCase(cfg=cfg, repository=mock_repo)  # ty:ignore[invalid-argument-type]

        usecase.execute()

        assert "staging_dir" in captured, "create() が呼ばれていない"
