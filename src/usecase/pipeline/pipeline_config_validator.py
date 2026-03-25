"""
Pipeline config 事前検証。

全 step の config を実行前にビルドし、OmegaConf の resolve を強制して
未解決の変数や Missing キーを全て検出する。

設計:
1. build_step_configs(): 各 step に usecase のデフォルト config をマージ
2. validate_pipeline_configs(): OmegaConf.to_container(resolve=True, throw_on_missing=True)
   で全キーの解決を試み、失敗した step のエラーを一括報告

config yaml 自体が「何が必要か」を定義しており、resolve 時に自動検出される。

時間計算量: O(S) — S: step 数
空間計算量: O(S)
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf

logger = logging.getLogger(__name__)

# usecase 名 → conf/usecase/ の yaml ファイル名
# NOTE: usecase yaml 名が usecase 名と一致しない場合のみここに登録する。
#       一致する場合は _resolve_yaml_name() のフォールバックで処理する。
_USECASE_TO_YAML: dict[str, str] = {
    "download_dataset": "download",
}


def _resolve_yaml_name(usecase_name: str) -> str:
    """usecase 名から conf/usecase/ の yaml ファイル名を解決する。"""
    return _USECASE_TO_YAML.get(usecase_name, usecase_name)


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
        result.append(DictConfig(merged))

    return result


def validate_pipeline_configs(step_configs: list[DictConfig]) -> None:
    """全 step の config を検証し、問題があれば一括でエラーを報告する。

    OmegaConf.to_container(resolve=True, throw_on_missing=True) で全キーの解決を試み、
    未解決の変数（${...}）や MISSING マーカーを自動検出する。
    Args:
        step_configs: build_step_configs() の戻り値

    Raises:
        PipelineConfigError: 1 つ以上の step に問題がある場合

    時間計算量: O(S)
    空間計算量: O(S)
    """
    errors: list[str] = []

    for i, cfg in enumerate(step_configs):
        step_num = i + 1
        usecase = str(cfg.get("usecase", "unknown"))

        try:
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        except MissingMandatoryValue as e:
            errors.append(
                f"Step {step_num} (usecase={usecase}):\n"
                f"  必須値が未設定です: {e}\n"
                f"  修正: pipeline yaml の step に値を追加するか、"
                f" conf/usecase/{_resolve_yaml_name(usecase)}.yaml"
                f" のデフォルト値を確認してください。"
            )
        except Exception as e:
            # InterpolationResolutionError 等: ${...} の変数が解決できない
            errors.append(
                f"Step {step_num} (usecase={usecase}):\n"
                f"  config の解決に失敗しました: {e}\n"
                f"  修正: 参照先の変数が定義されているか確認してください。"
            )

    if errors:
        msg = "Pipeline config 検証エラー — 実行前に以下を修正してください:\n\n" + "\n\n".join(
            errors
        )
        raise PipelineConfigError(msg)
