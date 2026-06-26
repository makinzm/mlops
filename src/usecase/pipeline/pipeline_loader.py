"""
pipeline 設定の検出・ロードユーティリティ。

conf/competition/{name}/pipeline/ 配下の yaml を検出し、
Hydra cfg とマージした DictConfig を返す。
"""

from pathlib import Path

from omegaconf import DictConfig

from src.usecase._recipe import load_single_recipe_cfg


def load_pipeline_recipe_cfg(cfg: DictConfig, conf_dir: Path) -> DictConfig:
    """competition の pipeline yaml をロードして cfg とマージした設定を返す。

    Args:
        cfg: Hydra の全体設定（competition.name と recipe が必要）。
        conf_dir: conf/ ルートディレクトリ。

    Returns:
        pipeline の DictConfig（cfg とマージ済み）。

    Raises:
        ValueError: recipe が指定されていない / yaml が見つからない場合。
    """
    return load_single_recipe_cfg(
        cfg,
        "pipeline",
        conf_dir,
        required=True,
        required_message=(
            "pipeline usecase には recipe= が必要です。\n"
            "例: uv run python -m src usecase=pipeline recipe=all_after_download"
        ),
    )
