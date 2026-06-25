"""
training パイプライン設定の検出・ロード。

conf/competition/{name}/training/ 配下の yaml を検出し、
Hydra cfg とマージした DictConfig リストを返す。
"""

from pathlib import Path

from omegaconf import DictConfig

from src.usecase._recipe import load_recipe_cfgs


def load_trainer_cfgs(cfg: DictConfig, conf_dir: Path) -> list[DictConfig]:
    """competition の training yaml をロードして cfg とマージした設定リストを返す。

    Args:
        cfg: Hydra の全体設定（competition.name が必要）。
             cfg.trainer_name が指定されていれば該当 yaml のみ返す。
        conf_dir: conf/ ルートディレクトリ。

    Returns:
        trainer ごとの DictConfig リスト（cfg とマージ済み）。

    Raises:
        ValueError: trainer が見つからない / training ディレクトリが空の場合。
    """
    # recipe= を優先し、旧パラメータ trainer_name= へのフォールバックで後方互換性を保つ
    return load_recipe_cfgs(
        cfg,
        "training",
        conf_dir,
        fallback_key="trainer_name",
        empty_dir_message=(
            f"training 設定が見つかりません: "
            f"{conf_dir / 'competition' / cfg.competition.name / 'training'}"
        ),
        label="trainer_name",
    )
