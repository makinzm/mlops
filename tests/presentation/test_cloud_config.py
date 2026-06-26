"""
presentation/cloud_config モジュールのテスト。

なぜこのテストが必要か:
  - _ensure_cloud_config, _resolve_manifest_path, _load_trainer_cfgs_safe は
    main.py から cloud_config.py に移動された。
  - 移動後も同じ動作を保証するために、既存テストに加えて
    モジュールの import パスが正しいことを検証する。
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


class TestEnsureCloudConfig:
    """_ensure_cloud_config() のテスト。"""

    def test_returns_cfg_when_infra_present(self) -> None:
        """infra が既に設定済みなら cfg をそのまま返すこと。"""
        from src.presentation.cloud_config import ensure_cloud_config

        cfg = OmegaConf.create({"infra": {"project": "test"}, "notification": {"type": "slack"}})
        result = ensure_cloud_config(cfg, "/dummy/conf")
        assert result.infra.project == "test"

    def test_merges_infra_yaml_when_missing(self, tmp_path: Path) -> None:
        """infra が None の場合、vertex.yaml をマージすること。"""
        from src.presentation.cloud_config import ensure_cloud_config

        conf_dir = tmp_path / "conf"
        infra_dir = conf_dir / "infra"
        infra_dir.mkdir(parents=True)
        (infra_dir / "vertex.yaml").write_text(
            "infra:\n  project: merged-project\n  region: us\n  staging_bucket: gs://b\n"
        )

        cfg = OmegaConf.create({"infra": None, "notification": {"type": "slack"}})
        result = ensure_cloud_config(cfg, str(conf_dir))
        assert result.infra.project == "merged-project"


class TestModuleImportPath:
    """移動後の import パスが正しいことを検証する。"""

    def test_resolve_manifest_path_importable(self) -> None:
        """_resolve_manifest_path が cloud_config から import できること。"""
        from src.presentation.cloud_config import resolve_manifest_path  # noqa: F401

    def test_load_trainer_cfgs_safe_importable(self) -> None:
        """_load_trainer_cfgs_safe が cloud_config から import できること。"""
        from src.presentation.cloud_config import load_trainer_cfgs_safe  # noqa: F401
