"""
recipe yaml の検出・ロード共通ユーティリティ。

conf/competition/{name}/{subdir}/ 配下の yaml を検出し、Hydra cfg とマージする。
trainer_loader / inference_loader / preprocessing.pipeline_loader /
pipeline.pipeline_loader / notebook_loader の共通実装をここに集約する。
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def _merge(cfg: DictConfig, yaml_path: Path) -> DictConfig:
    """to_container で plain dict に変換してから merge し、Hydra の struct モード制約を回避する。"""
    base = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    return DictConfig(OmegaConf.merge(base, OmegaConf.load(yaml_path)))


def _not_found_error(label: str, value: str, competition_name: str, target_dir: Path) -> ValueError:
    available = [f.stem for f in sorted(target_dir.glob("*.yaml"))]
    return ValueError(
        f"{label}='{value}' が見つかりません"
        f"（competition: {competition_name}）。"
        f" 利用可能: {available}"
    )


def load_recipe_cfgs(
    cfg: DictConfig,
    subdir: str,
    conf_dir: Path,
    *,
    fallback_key: str | None = None,
    empty_dir_message: str | None = None,
    label: str = "recipe",
) -> list[DictConfig]:
    """recipe 指定時は単一 yaml、未指定時は subdir 配下の全 yaml をロードして cfg とマージする。

    Args:
        cfg: Hydra の全体設定（competition.name が必要）。cfg.recipe を優先して参照する。
        subdir: conf/competition/{name}/ 配下のサブディレクトリ名（"training" 等）。
        conf_dir: conf/ ルートディレクトリ。
        fallback_key: cfg.recipe が未指定のときに参照するレガシーキー名。
        empty_dir_message: subdir 配下に yaml が無いときのエラーメッセージ。省略時は既定文を使う。
        label: 見つからないときのエラーメッセージに使うキー名（"recipe" 等）。

    Returns:
        マージ済み DictConfig のリスト。

    Raises:
        ValueError: 指定した recipe が見つからない、または subdir が空の場合。
    """
    competition_name: str = cfg.competition.name
    recipe: str | None = cfg.get("recipe")
    if recipe is None and fallback_key is not None:
        recipe = cfg.get(fallback_key)

    target_dir = conf_dir / "competition" / competition_name / subdir

    if recipe is not None:
        target = target_dir / f"{recipe}.yaml"
        if not target.exists():
            raise _not_found_error(label, recipe, competition_name, target_dir)
        yaml_files = [target]
    else:
        yaml_files = sorted(target_dir.glob("*.yaml"))
        if not yaml_files:
            raise ValueError(empty_dir_message or f"{subdir} 設定が見つかりません: {target_dir}")

    return [_merge(cfg, f) for f in yaml_files]


def load_single_recipe_cfg(
    cfg: DictConfig,
    subdir: str,
    conf_dir: Path,
    *,
    required: bool = False,
    required_message: str | None = None,
    label: str = "recipe",
) -> DictConfig:
    """recipe 指定時のみ単一 yaml をロードして cfg とマージする。

    recipe 未指定時は required に応じて raise するか cfg をそのまま返す（passthrough）。

    Args:
        cfg: Hydra の全体設定（competition.name と recipe を参照する）。
        subdir: conf/competition/{name}/ 配下のサブディレクトリ名。
        conf_dir: conf/ ルートディレクトリ。
        required: True の場合、recipe 未指定時に ValueError を送出する。
        required_message: required=True かつ recipe 未指定時のエラーメッセージ。
        label: 見つからないときのエラーメッセージに使うキー名。

    Returns:
        recipe 指定時はマージ済み DictConfig。未指定・required=False のときは cfg をそのまま返す。

    Raises:
        ValueError: required=True で recipe 未指定、または指定した recipe が見つからない場合。
    """
    recipe: str | None = cfg.get("recipe")
    if recipe is None:
        if required:
            raise ValueError(required_message or f"{subdir} usecase には recipe= が必要です。")
        return cfg

    competition_name: str = cfg.competition.name
    target_dir = conf_dir / "competition" / competition_name / subdir
    target = target_dir / f"{recipe}.yaml"
    if not target.exists():
        raise _not_found_error(label, recipe, competition_name, target_dir)
    return _merge(cfg, target)
