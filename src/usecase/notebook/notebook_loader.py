"""
notebook 設定の検出・ロードユーティリティ。

conf/competition/{name}/notebook/ 配下の yaml を検出し、
Hydra cfg とマージした DictConfig を返す。

push_notebook usecase yaml は routing-only（usecase, output_dir のみ）にしているため、
直接 CLI 実行時（pipeline 経由でない場合）は cfg.recipe から competition 固有の
notebook 設定（competition, kernel_slug, src_dataset 等）をこのモジュールでロードする。
pipeline 経由（step で notebook: を直接 dict 指定）の場合は recipe が無いため、
cfg をそのまま返す（既存の pipeline 実行を壊さない）。
"""

from pathlib import Path

from omegaconf import DictConfig

from src.usecase._recipe import load_single_recipe_cfg


def load_notebook_recipe_cfg(cfg: DictConfig, conf_dir: Path) -> DictConfig:
    """competition の notebook yaml をロードして cfg とマージした設定を返す。

    Args:
        cfg: Hydra の全体設定（competition.name が必要）。
             cfg.recipe が指定されていれば該当 yaml をマージする。
        conf_dir: conf/ ルートディレクトリ。

    Returns:
        notebook 設定をマージした DictConfig。recipe 未指定時は cfg をそのまま返す。

    Raises:
        ValueError: recipe を指定したが該当 yaml が見つからない場合。
    """
    return load_single_recipe_cfg(cfg, "notebook", conf_dir, required=False)
