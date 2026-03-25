"""
PipelineUseCase — 複数 usecase を順次実行するパイプライン。

cfg.steps を順番にループし、step.usecase に応じて
注入された runner 関数を呼ぶ。

実行前に全 step の config を事前検証する:
1. 各 step に usecase のデフォルト config をマージ
2. 必須キーの存在を全 step 分チェック
3. 問題があれば全エラーを一括で表示して停止

設計上の注意:
- PipelineUseCase 自身はインフラに依存しない。
  runner 関数を外部（main.py）から DI することで、
  テスト時は Mock runner に差し替えられる。
- fail-fast: 1ステップが例外を上げたら後続ステップを実行せずに例外を伝播する。
- 全パスは各 step の cfg で管理する（Hydra Config）。
- runner は **extra_runners で拡張可能。新 usecase 追加時は main.py 側に追加する。

新しい usecase を追加する場合:
1. main.py に _run_xxx() 関数を追加
2. PipelineUseCase の初期化時に run_xxx=_run_xxx を渡す
3. pipeline_config_validator.py の _USECASE_TO_YAML と _REQUIRED_KEYS に追加
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.usecase.pipeline.pipeline_config_validator import (
    build_step_configs,
    validate_pipeline_configs,
)

logger = logging.getLogger(__name__)


class PipelineUseCase:
    """steps に定義された usecase を順次実行する。"""

    def __init__(
        self,
        run_preprocess: Callable[[DictConfig], None],
        run_train: Callable[[DictConfig], None],
        run_inference: Callable[[DictConfig], None],
        conf_dir: Path | None = None,
        **extra_runners: Callable[[DictConfig], None],
    ) -> None:
        # extra_runners のキーは "run_<name>" 形式で渡されるため、
        # "run_" プレフィックスを除去して step.usecase の値に合わせる。
        # 例: run_remote_train → remote_train
        normalized_extras = {
            (k[len("run_") :] if k.startswith("run_") else k): v for k, v in extra_runners.items()
        }
        self._runners: dict[str, Callable[[DictConfig], None]] = {
            "preprocess": run_preprocess,
            "train": run_train,
            "inference": run_inference,
            **normalized_extras,
        }
        self._conf_dir = conf_dir

    def run(self, cfg: DictConfig) -> None:
        """cfg.steps の順番に各 usecase を実行する。

        実行前に全 step の config を事前検証する。
        問題があれば PipelineConfigError を送出して全エラーを一括表示する。

        Args:
            cfg: PipelineUseCase 用の DictConfig。
                 cfg.steps は step ごとの設定リスト（usecase, recipe 等）。

        Raises:
            PipelineConfigError: config に必須キーが不足している場合
            ValueError: step.usecase が未知の値の場合
            Exception: 各 runner が上げる例外をそのまま伝播（fail-fast）
        """
        steps = cfg.get("steps", [])

        # 1. 全 step の usecase が登録されているか確認
        for step in steps:
            step_usecase: str = str(step.get("usecase", ""))
            if step_usecase not in self._runners:
                raise ValueError(
                    f"step.usecase='{step_usecase}' は未登録です。"
                    f" 登録済み: {list(self._runners.keys())}"
                )

        # 2. 全 step の config を事前ビルド・検証
        if self._conf_dir is not None:
            step_configs = build_step_configs(cfg, self._conf_dir)
            validate_pipeline_configs(step_configs)
            logger.info(f"Pipeline config 検証 OK: {len(step_configs)} step(s)")
        else:
            step_configs = None

        # 3. 実行
        for i, step in enumerate(steps):
            step_usecase = str(step.get("usecase", ""))

            if step_configs is not None:
                merged = step_configs[i]
            else:
                # conf_dir が未指定の場合は従来の動作（後方互換）
                base_cfg_ns = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                step_cfg_ns = OmegaConf.create(OmegaConf.to_container(step, resolve=True))
                merged = OmegaConf.merge(base_cfg_ns, step_cfg_ns)

            logger.info(f"Pipeline step {i + 1}/{len(steps)}: {step_usecase}")
            self._runners[step_usecase](DictConfig(merged))
