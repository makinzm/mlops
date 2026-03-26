"""UseCase ディスパッチレジストリ。

main.py の巨大な if/elif チェーンを辞書ベースの dispatch に置き換える。
新しい usecase を追加する場合は RUNNERS に登録するだけでよい。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig

from src.presentation.runners import (
    run_automatically_eda,
    run_create_source_dataset,
    run_download,
    run_gradcam,
    run_inference,
    run_pipeline,
    run_preprocess,
    run_push_notebook,
    run_remote_train,
    run_train,
    run_update_source_dataset,
    run_vertex_download,
    run_vertex_submit,
)

Runner = Callable[[DictConfig, Any], None]

RUNNERS: dict[str, Runner] = {
    "download_dataset": run_download,
    "automatically_eda": run_automatically_eda,
    "preprocess": run_preprocess,
    "train": run_train,
    "inference": run_inference,
    "remote_train": run_remote_train,
    "vertex_submit": run_vertex_submit,
    "vertex_download": run_vertex_download,
    "pipeline": run_pipeline,
    "push_notebook": run_push_notebook,
    "create_source_dataset": run_create_source_dataset,
    "update_source_dataset": run_update_source_dataset,
    "gradcam": run_gradcam,
}


def dispatch(
    usecase_name: str,
    cfg: DictConfig,
    logger: Any,
    *,
    overrides: dict[str, Runner] | None = None,
) -> None:
    """usecase_name に対応する runner を実行する。

    Args:
        usecase_name: 実行する usecase の名前
        cfg: Hydra DictConfig
        logger: ロガー
        overrides: テスト用の runner 上書き辞書

    Raises:
        ValueError: 未知の usecase の場合
    """
    runners = {**RUNNERS, **(overrides or {})}
    runner = runners.get(usecase_name)
    if runner is None:
        raise ValueError(f"Unknown usecase: {usecase_name!r}. Supported: {sorted(runners.keys())}")
    runner(cfg, logger)
