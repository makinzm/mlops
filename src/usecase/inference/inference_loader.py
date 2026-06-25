"""
inference 設定の検出・ロードユーティリティ。

conf/competition/{name}/inference/ 配下の yaml を検出し、
Hydra cfg とマージした DictConfig リストを返す。
"""

from pathlib import Path

from omegaconf import DictConfig

from src.usecase._recipe import load_recipe_cfgs


def load_inference_cfgs(cfg: DictConfig, conf_dir: Path) -> list[DictConfig]:
    """competition の inference yaml をロードして cfg とマージした設定リストを返す。

    Args:
        cfg: Hydra の全体設定（competition.name が必要）。
             cfg.recipe が指定されていれば該当 yaml のみ返す。
        conf_dir: conf/ ルートディレクトリ。

    Returns:
        inference ごとの DictConfig リスト（cfg とマージ済み）。

    Raises:
        ValueError: recipe が見つからない / inference ディレクトリが空の場合。
    """
    return load_recipe_cfgs(
        cfg,
        "inference",
        conf_dir,
        empty_dir_message=(
            f"inference 設定が見つかりません: "
            f"{conf_dir / 'competition' / cfg.competition.name / 'inference'}"
        ),
    )
