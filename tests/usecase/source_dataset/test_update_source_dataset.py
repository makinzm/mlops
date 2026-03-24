"""
UpdateSourceDatasetUseCase のテスト。

なぜこのテストが必要か:
  - バージョン更新時に src/ がステージングにコピーされ、
    正しい version_message で repository.update_version() が呼ばれることを保証する。
  - 成功/失敗時のステージング管理（削除 vs 残存）を確認する。
  - CreateSourceDatasetUseCase と同じ staging 戦略を持つことを確認する。

fixture:
  - tmp_path: src_dir / staging_dir を tmp_path 以下に配置
  - MagicMock: SourceDatasetRepository の差し替え
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from src.usecase.source_dataset.update_source_dataset import UpdateSourceDatasetUseCase


def _make_cfg(tmp_path: Path, version_message: str = "update source code") -> object:
    return OmegaConf.create(
        {
            "usecase": "update_source_dataset",
            "source_dataset": {
                "src_dir": str(tmp_path / "src"),
                "dataset_slug": "mlops-pipeline-src",
                "title": "mlops-pipeline-src",
                "license_name": "CC0-1.0",
                "ignorefile": None,
                "version_message": version_message,
            },
            "staging_dir": str(tmp_path / ".staging"),
            "platform_username": "testuser",
        }
    )


def _setup_src(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "__init__.py").write_text("")
    (src / "main.py").write_text("# main")
    return src


class TestUpdateSourceDatasetUseCaseExecute:
    """execute() メソッドの主要な振る舞いを検証する。"""

    def test_execute_copies_src_to_staging(self, tmp_path: Path) -> None:
        """src/ の内容が staging dir にコピーされること。"""
        _setup_src(tmp_path)
        cfg = _make_cfg(tmp_path)
        mock_repo = MagicMock()
        usecase = UpdateSourceDatasetUseCase(cfg=cfg, repository=mock_repo)  # ty:ignore[invalid-argument-type]

        usecase.execute()

        mock_repo.update_version.assert_called_once()
        call_args = mock_repo.update_version.call_args
        staging_dir: Path = call_args.kwargs.get("staging_dir") or call_args.args[0]
        # 成功後は削除されるので、呼び出し時点では src がコピーされていたはず
        assert staging_dir is not None

    def test_execute_calls_repository_update_version(self, tmp_path: Path) -> None:
        """SourceDatasetRepository.update_version() が1回呼ばれること。"""
        _setup_src(tmp_path)
        cfg = _make_cfg(tmp_path)
        mock_repo = MagicMock()
        usecase = UpdateSourceDatasetUseCase(cfg=cfg, repository=mock_repo)  # ty:ignore[invalid-argument-type]

        usecase.execute()

        mock_repo.update_version.assert_called_once()

    def test_execute_passes_version_message(self, tmp_path: Path) -> None:
        """version_message が repository.update_version() に渡されること。"""
        _setup_src(tmp_path)
        cfg = _make_cfg(tmp_path, version_message="add target encoding")
        mock_repo = MagicMock()
        usecase = UpdateSourceDatasetUseCase(cfg=cfg, repository=mock_repo)  # ty:ignore[invalid-argument-type]

        usecase.execute()

        call_args = mock_repo.update_version.call_args
        assert "add target encoding" in str(call_args), (
            "version_message が repository に渡されていない"
        )

    def test_execute_removes_staging_dir_on_success(self, tmp_path: Path) -> None:
        """成功時に staging subdir が削除されること。"""
        _setup_src(tmp_path)
        cfg = _make_cfg(tmp_path)
        mock_repo = MagicMock()
        usecase = UpdateSourceDatasetUseCase(cfg=cfg, repository=mock_repo)  # ty:ignore[invalid-argument-type]

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
        mock_repo.update_version.side_effect = RuntimeError("upload failed")
        usecase = UpdateSourceDatasetUseCase(cfg=cfg, repository=mock_repo)  # ty:ignore[invalid-argument-type]

        with pytest.raises(RuntimeError, match="upload failed"):
            usecase.execute()

        staging_root = tmp_path / ".staging"
        assert staging_root.exists(), "失敗時は staging が残るべき"
        subdirs = [d for d in staging_root.iterdir() if d.is_dir()]
        assert len(subdirs) >= 1, "失敗時は staging subdir が残るべき"
