"""
PipelineUseCase — 複数 usecase を順次実行するパイプライン。

cfg.steps を順番にループし、step.usecase に応じて
注入された runner 関数を呼ぶ。

設計上の注意:
- PipelineUseCase 自身はインフラに依存しない。
  runner 関数を外部（main.py）から DI することで、
  テスト時は Mock runner に差し替えられる。
- fail-fast: 1ステップが例外を上げたら後続ステップを実行せずに例外を伝播する。
- 全パスは各 step の cfg で管理する（Hydra Config）。
- runner は **extra_runners で拡張可能。新 usecase 追加時は main.py 側に追加する。
"""

from __future__ import annotations

from collections.abc import Callable

from omegaconf import DictConfig, OmegaConf


class PipelineUseCase:
    """steps に定義された usecase を順次実行する。"""

    def __init__(
        self,
        run_preprocess: Callable[[DictConfig], None],
        run_train: Callable[[DictConfig], None],
        run_inference: Callable[[DictConfig], None],
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

    def run(self, cfg: DictConfig) -> None:
        """cfg.steps の順番に各 usecase を実行する。

        Args:
            cfg: PipelineUseCase 用の DictConfig。
                 cfg.steps は step ごとの設定リスト（usecase, recipe 等）。

        Raises:
            ValueError: step.usecase が未知の値の場合
            Exception: 各 runner が上げる例外をそのまま伝播（fail-fast）
        """
        steps = cfg.get("steps", [])
        for step in steps:
            step_usecase: str = str(step.get("usecase", ""))
            if step_usecase not in self._runners:
                raise ValueError(
                    f"step.usecase='{step_usecase}' は未登録です。"
                    f" 登録済み: {list(self._runners.keys())}"
                )

            # Hydra の struct モード制約を回避するため to_container で plain dict に変換
            # OmegaConf.merge は struct モードの DictConfig に未定義キーを追加できないため、
            # 一度 non-struct な DictConfig に変換してからマージする
            base_cfg_ns = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            step_cfg_ns = OmegaConf.create(OmegaConf.to_container(step, resolve=True))
            merged = OmegaConf.merge(base_cfg_ns, step_cfg_ns)

            self._runners[step_usecase](DictConfig(merged))
