"""
preprocess パイプライン設定の検出・ロードユーティリティ。

conf/competition/{name}/preprocess/ 配下の yaml を検出し、
Hydra cfg とマージした DictConfig リストを返す。
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


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
    competition_name: str = cfg.competition.name
    # recipe= を優先し、旧パラメータ pipeline= へのフォールバックで後方互換性を保つ
    pipeline: str | None = cfg.get("recipe", cfg.get("pipeline"))

    preprocess_dir = conf_dir / "competition" / competition_name / "preprocess"

    if pipeline is not None:
        target = preprocess_dir / f"{pipeline}.yaml"
        if not target.exists():
            available = [f.stem for f in sorted(preprocess_dir.glob("*.yaml"))]
            raise ValueError(
                f"pipeline='{pipeline}' が見つかりません"
                f"（competition: {competition_name}）。"
                f" 利用可能: {available}"
            )
        yaml_files = [target]
    else:
        yaml_files = sorted(preprocess_dir.glob("*.yaml"))
        if not yaml_files:
            raise ValueError(f"前処理設定が見つかりません: {preprocess_dir}")

    # Hydra の struct モード制約を回避するため to_container で plain dict に変換してからマージ
    base = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    return [DictConfig(OmegaConf.merge(base, OmegaConf.load(f))) for f in yaml_files]
