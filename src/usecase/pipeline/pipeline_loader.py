"""
pipeline 設定の検出・ロードユーティリティ。

conf/competition/{name}/pipeline/ 配下の yaml を検出し、
Hydra cfg とマージした DictConfig を返す。
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


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
    competition_name: str = cfg.competition.name
    recipe: str | None = cfg.get("recipe")

    pipeline_dir = conf_dir / "competition" / competition_name / "pipeline"

    if recipe is None:
        raise ValueError(
            "pipeline usecase には recipe= が必要です。\n"
            "例: uv run python -m src usecase=pipeline recipe=all_after_download"
        )

    target = pipeline_dir / f"{recipe}.yaml"
    if not target.exists():
        available = [f.stem for f in sorted(pipeline_dir.glob("*.yaml"))]
        raise ValueError(
            f"recipe='{recipe}' が見つかりません"
            f"（competition: {competition_name}）。"
            f" 利用可能: {available}"
        )

    # Hydra の struct モード制約を回避するため to_container で plain dict に変換してからマージ
    base = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    return DictConfig(OmegaConf.merge(base, OmegaConf.load(target)))
