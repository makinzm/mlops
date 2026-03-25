"""
Pipeline config 事前検証のテスト。

なぜこのテストが必要か:
  - Pipeline 実行前に全 step の config を検証し、足りないキーを一括でエラー表示する。
  - OmegaConf の resolve を強制して Missing や未解決変数を自動検出する。
  - ハードコードリストなしで OmegaConf resolve により自動検出されることを確認する。

時間計算量: O(S) — S: step 数
空間計算量: O(S)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import MISSING, DictConfig, OmegaConf

from src.usecase.pipeline.pipeline_config_validator import (
    PipelineConfigError,
    build_step_configs,
    validate_pipeline_configs,
)


@pytest.fixture
def conf_dir(tmp_path: Path) -> Path:
    """最小限の conf ディレクトリを作成する。"""
    usecase_dir = tmp_path / "usecase"
    usecase_dir.mkdir()
    (usecase_dir / "download.yaml").write_text(
        "# @package _global_\n"
        "usecase: download_dataset\n"
        "output_dir: data/raw/${competition.name}\n"
        "unzip: true\n"
        "force: true\n"
        "source: kaggle\n"
        "kaggle:\n"
        "  mode: competition\n"
        "  competition: ${competition.name}\n"
        "  dataset: null\n"
        "seed: 42\n"
    )
    (usecase_dir / "train.yaml").write_text("# @package _global_\nusecase: train\n")
    (usecase_dir / "inference.yaml").write_text("# @package _global_\nusecase: inference\n")
    (usecase_dir / "preprocess.yaml").write_text("# @package _global_\nusecase: preprocess\n")
    return tmp_path


class TestBuildStepConfigs:
    def test_loads_usecase_defaults(self, conf_dir: Path) -> None:
        """step config に usecase のデフォルト値がマージされること。"""
        pipeline_cfg = OmegaConf.create(
            {
                "competition": {"name": "test_comp"},
                "steps": [{"usecase": "download_dataset"}],
            }
        )
        step_configs = build_step_configs(pipeline_cfg, conf_dir)
        assert len(step_configs) == 1
        cfg = step_configs[0]
        assert cfg.get("unzip") is True
        assert cfg.get("force") is True
        assert cfg.get("source") == "kaggle"

    def test_step_overrides_take_precedence(self, conf_dir: Path) -> None:
        """step の設定が usecase デフォルトより優先されること。"""
        pipeline_cfg = OmegaConf.create(
            {
                "competition": {"name": "test_comp"},
                "steps": [{"usecase": "download_dataset", "force": False}],
            }
        )
        step_configs = build_step_configs(pipeline_cfg, conf_dir)
        assert step_configs[0].get("force") is False


class TestValidatePipelineConfigs:
    def test_valid_config_passes(self, conf_dir: Path) -> None:
        """正しい config ではエラーが出ないこと。"""
        pipeline_cfg = OmegaConf.create(
            {
                "competition": {"name": "test_comp"},
                "steps": [{"usecase": "download_dataset"}],
            }
        )
        step_configs = build_step_configs(pipeline_cfg, conf_dir)
        validate_pipeline_configs(step_configs)

    def test_missing_mandatory_value_detected(self) -> None:
        """MISSING マーカーがある場合にエラーを検出すること。"""
        step_configs = [
            DictConfig({"usecase": "train", "required_field": MISSING}),
        ]
        with pytest.raises(PipelineConfigError) as exc_info:
            validate_pipeline_configs(step_configs)
        assert "Step 1" in str(exc_info.value)
        assert "必須値が未設定" in str(exc_info.value)

    def test_unresolved_interpolation_detected(self) -> None:
        """未解決の ${...} 変数がある場合にエラーを検出すること。"""
        step_configs = [
            DictConfig({"usecase": "download_dataset", "path": "${nonexistent_var}"}),
        ]
        with pytest.raises(PipelineConfigError) as exc_info:
            validate_pipeline_configs(step_configs)
        assert "Step 1" in str(exc_info.value)

    def test_multiple_errors_reported_at_once(self) -> None:
        """複数 step のエラーが一括で報告されること。"""
        step_configs = [
            DictConfig({"usecase": "train", "x": MISSING}),
            DictConfig({"usecase": "inference", "y": MISSING}),
        ]
        with pytest.raises(PipelineConfigError) as exc_info:
            validate_pipeline_configs(step_configs)
        error_msg = str(exc_info.value)
        assert "Step 1" in error_msg
        assert "Step 2" in error_msg

    def test_valid_config_with_null_passes(self) -> None:
        """null 値は MISSING ではないのでエラーにならないこと。"""
        step_configs = [
            DictConfig({"usecase": "train", "recipe": None}),
        ]
        validate_pipeline_configs(step_configs)
