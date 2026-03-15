"""
training パイプライン設定の検出・ロードと Trainer ファクトリ。

conf/competition/{name}/training/ 配下の yaml を検出し、
Hydra cfg とマージした DictConfig リストを返す。
またcfg.trainer.type に基づいて Trainer 実装を返す。
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.domain.model.trainer import Trainer


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
    competition_name: str = cfg.competition.name
    trainer_name: str | None = cfg.get("trainer_name")

    training_dir = conf_dir / "competition" / competition_name / "training"

    if trainer_name is not None:
        target = training_dir / f"{trainer_name}.yaml"
        if not target.exists():
            available = [f.stem for f in sorted(training_dir.glob("*.yaml"))]
            raise ValueError(
                f"trainer_name='{trainer_name}' が見つかりません"
                f"（competition: {competition_name}）。"
                f" 利用可能: {available}"
            )
        yaml_files = [target]
    else:
        yaml_files = sorted(training_dir.glob("*.yaml"))
        if not yaml_files:
            raise ValueError(f"training 設定が見つかりません: {training_dir}")

    # Hydra の struct モード制約を回避するため to_container で plain dict に変換してからマージ
    base = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    return [DictConfig(OmegaConf.merge(base, OmegaConf.load(f))) for f in yaml_files]


def resolve_trainer(cfg: DictConfig) -> Trainer:
    """cfg.trainer.type に基づいて Trainer 実装を返すファクトリ。

    Args:
        cfg: trainer.type キーを持つ DictConfig。

    Returns:
        Trainer Protocol を満たすオブジェクト。

    Raises:
        ValueError: 未登録の trainer type の場合。
    """
    trainer_type: str = cfg.trainer.type

    if trainer_type == "lgbm":
        from src.infrastructure.trainer.lgbm_trainer import LightGBMTrainer

        return LightGBMTrainer(cfg)

    raise ValueError(f"trainer.type='{trainer_type}' は未登録です。 登録済み: ['lgbm']")
