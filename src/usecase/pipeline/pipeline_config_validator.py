"""
Pipeline config 事前検証。

全 step の config を実行前にビルド・検証し、
足りないキーがあれば全て列挙してから PipelineConfigError を送出する。

設計:
1. build_step_configs(): 各 step に usecase のデフォルト config をマージ
2. validate_pipeline_configs(): 必須キーの存在チェック

時間計算量: O(S * K) — S: step 数, K: チェックするキー数
空間計算量: O(S)
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# usecase 名 → conf/usecase/ の yaml ファイル名
# NOTE: mille の name_deny でベンダー名を含む文字列リテラルが禁止されているため、
#       usecase yaml 名が usecase 名と一致しない場合のみここに登録する。
#       一致する場合は _resolve_yaml_name() のフォールバックで処理する。
_USECASE_TO_YAML: dict[str, str] = {
    "download_dataset": "download",
}


def _resolve_yaml_name(usecase_name: str) -> str:
    """usecase 名から conf/usecase/ の yaml ファイル名を解決する。"""
    return _USECASE_TO_YAML.get(usecase_name, usecase_name)


# usecase 名 → 必須キーのリスト
_REQUIRED_KEYS: dict[str, list[str]] = {
    "download_dataset": ["output_dir", "source"],
    "preprocess": ["competition"],
    "train": ["competition"],
    "inference": ["competition"],
    "remote_train": ["competition"],
    "push_notebook": ["notebook"],
    "update_source_dataset": ["source_dataset"],
    "create_source_dataset": ["source_dataset"],
    "gradcam": ["model_path", "image_dir", "output_dir"],
}


class PipelineConfigError(Exception):
    """Pipeline config の検証エラー。全てのエラーを一括で報告する。"""

    pass


def build_step_configs(
    pipeline_cfg: DictConfig,
    conf_dir: Path,
) -> list[DictConfig]:
    """各 step に usecase のデフォルト config をマージした DictConfig リストを返す。

    マージ順序: pipeline_base → usecase_defaults → step_overrides
    step_overrides が最も優先度が高い。

    Args:
        pipeline_cfg: Pipeline の DictConfig（steps を含む）
        conf_dir: conf/ ルートディレクトリ

    Returns:
        各 step のマージ済み DictConfig リスト

    時間計算量: O(S) — S: step 数
    空間計算量: O(S)
    """
    steps = pipeline_cfg.get("steps", [])
    result: list[DictConfig] = []

    for step in steps:
        step_usecase: str = str(step.get("usecase", ""))

        # 1. pipeline の base config（steps を除く）
        base_dict = OmegaConf.to_container(pipeline_cfg, resolve=False)
        if isinstance(base_dict, dict):
            base_dict.pop("steps", None)
        base_cfg = OmegaConf.create(base_dict)

        # 2. usecase のデフォルト config をロード
        yaml_name = _resolve_yaml_name(step_usecase)
        usecase_yaml = conf_dir / "usecase" / f"{yaml_name}.yaml"
        if usecase_yaml.exists():
            usecase_defaults = OmegaConf.load(usecase_yaml)
        else:
            usecase_defaults = OmegaConf.create({})

        # 3. step の override config
        step_cfg = OmegaConf.create(OmegaConf.to_container(step, resolve=False))

        # マージ: base → usecase_defaults → step_overrides
        merged = OmegaConf.merge(base_cfg, usecase_defaults, step_cfg)

        # OmegaConf 変数を解決
        try:
            resolved = OmegaConf.create(OmegaConf.to_container(merged, resolve=True))
        except Exception:
            # 解決できない変数がある場合は未解決のまま返す（検証で捕捉する）
            resolved = DictConfig(OmegaConf.to_container(merged, resolve=False))

        result.append(DictConfig(resolved))

    return result


def validate_pipeline_configs(step_configs: list[DictConfig]) -> None:
    """全 step の config を検証し、問題があれば一括でエラーを報告する。

    Args:
        step_configs: build_step_configs() の戻り値

    Raises:
        PipelineConfigError: 1 つ以上の step に問題がある場合

    時間計算量: O(S * K)
    空間計算量: O(S)
    """
    errors: list[str] = []

    for i, cfg in enumerate(step_configs):
        step_num = i + 1
        usecase = str(cfg.get("usecase", "unknown"))
        required = _REQUIRED_KEYS.get(usecase, [])

        step_errors: list[str] = []
        for key in required:
            if cfg.get(key) is None:
                step_errors.append(
                    f"  - '{key}' が未設定です。"
                    f" conf/usecase/{_resolve_yaml_name(usecase)}.yaml"
                    f" のデフォルト値を確認してください。"
                )

        if step_errors:
            errors.append(f"Step {step_num} (usecase={usecase}):\n" + "\n".join(step_errors))

    if errors:
        msg = "Pipeline config 検証エラー — 実行前に以下を修正してください:\n\n" + "\n\n".join(
            errors
        )
        raise PipelineConfigError(msg)
