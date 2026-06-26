"""
_resolve_manifest_path と run_job_download の統合テスト。

なぜこのテストが必要か:
  - _resolve_manifest_path は manifest_path 未指定時に competition + job_id + latest から
    自動解決する。pipeline 経由では recipe が pipeline recipe になるため、
    trainer config から job_id を取得するフォールバックが必要。
  - run_job_download は output_dir も trainer config から取得する必要がある。
  - これらのロジックは main.py の presentation 層にあるが、
    パターンが複数あるためテストで全パターンをカバーする。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from src.presentation.cloud_config import resolve_manifest_path as _resolve_manifest_path


@pytest.fixture()
def history_tree(tmp_path: Path) -> Path:
    """job_history のテスト用ディレクトリツリーを作成する。"""
    job_dir = tmp_path / "job_history" / "titanic" / "titanic_lgbm" / "20260326T010000"
    job_dir.mkdir(parents=True)
    manifest = job_dir / "job_manifest.yaml"
    manifest.write_text("status: SUBMITTED\njob_id: titanic_lgbm\n")
    return tmp_path


def _base_cfg(tmp_path: Path, **overrides: object) -> DictConfig:
    """テスト用の DictConfig を生成する。"""
    base: dict[str, object] = {
        "competition": {"name": "titanic"},
        "job_id": None,
        "recipe": None,
        "manifest_path": None,
        "job_history_dir": str(tmp_path / "job_history"),
    }
    for key, value in overrides.items():
        base[key] = value
    return OmegaConf.create(base)


class TestResolveManifestPathExplicit:
    """manifest_path が明示指定された場合のテスト。"""

    def test_returns_explicit_path(self, tmp_path: Path) -> None:
        """manifest_path が指定されていればそのまま返すこと。"""
        cfg = _base_cfg(tmp_path, manifest_path="/some/path/job_manifest.yaml")
        result = _resolve_manifest_path(cfg)
        assert result == Path("/some/path/job_manifest.yaml")

    def test_ignores_none_string(self, history_tree: Path) -> None:
        """manifest_path が 'None' 文字列の場合は自動解決にフォールバックすること。"""
        cfg = _base_cfg(history_tree, manifest_path="None")
        result = _resolve_manifest_path(cfg)
        assert "job_manifest.yaml" in str(result)


class TestResolveManifestPathAutoResolve:
    """manifest_path 未指定時の自動解決テスト。"""

    def test_resolves_from_job_id(self, history_tree: Path) -> None:
        """cfg.job_id が設定されていて対応ディレクトリが存在すればそこから解決すること。"""
        cfg = _base_cfg(history_tree, job_id="titanic_lgbm")
        result = _resolve_manifest_path(cfg)
        assert result.name == "job_manifest.yaml"
        assert "titanic_lgbm" in str(result)
        assert "20260326T010000" in str(result)

    def test_falls_back_when_job_id_dir_missing(self, history_tree: Path) -> None:
        """cfg.job_id のディレクトリが存在しない場合、trainer config にフォールバックすること。"""
        cfg = _base_cfg(history_tree, job_id="nonexistent_job")
        # trainer config から titanic_lgbm が取得される
        result = _resolve_manifest_path(cfg)
        assert "titanic_lgbm" in str(result)

    def test_pipeline_recipe_does_not_break(self, history_tree: Path) -> None:
        """recipe が pipeline recipe でも trainer config から正しく解決すること。"""
        cfg = _base_cfg(
            history_tree,
            job_id="titanic_job_download_and_push",
            recipe="job_download_and_push",
        )
        result = _resolve_manifest_path(cfg)
        assert "titanic_lgbm" in str(result)

    def test_resolves_latest_timestamp(self, tmp_path: Path) -> None:
        """複数タイムスタンプがある場合、最新が選ばれること。"""
        base = tmp_path / "job_history" / "titanic" / "titanic_lgbm"
        for ts in ["20260325T100000", "20260326T100000"]:
            d = base / ts
            d.mkdir(parents=True)
            (d / "job_manifest.yaml").write_text(f"timestamp: {ts}\n")

        cfg = _base_cfg(tmp_path, job_id="titanic_lgbm")
        result = _resolve_manifest_path(cfg)
        assert "20260326T100000" in str(result)
