"""
preprocess パイプライン設定の検出・ロードユーティリティ。

conf/competition/{name}/preprocess/ 配下の yaml を検出し、
Hydra cfg とマージした DictConfig リストを返す。
"""

from pathlib import Path

from omegaconf import DictConfig

from src.usecase._recipe import load_recipe_cfgs


def load_pipeline_cfgs(cfg: DictConfig, conf_dir: Path) -> list[DictConfig]:
    """competition の preprocess yaml をロードして cfg とマージした設定リストを返す。

    Args:
        cfg: Hydra の全体設定（competition.name が必要）。
        conf_dir: conf/ ルートディレクトリ。

    Returns:
        pipeline ごとの DictConfig リスト（cfg とマージ済み）。

    Raises:
        ValueError: pipeline が見つからない / preprocess ディレクトリが空の場合。
    """
    # recipe= を優先し、旧パラメータ pipeline= へのフォールバックで後方互換性を保つ
    return load_recipe_cfgs(
        cfg,
        "preprocess",
        conf_dir,
        fallback_key="pipeline",
        empty_dir_message=(
            f"前処理設定が見つかりません: "
            f"{conf_dir / 'competition' / cfg.competition.name / 'preprocess'}"
        ),
        label="pipeline",
    )
