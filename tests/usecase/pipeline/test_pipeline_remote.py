"""
PipelineUseCase の job_train / update_source_dataset / push_notebook 対応テスト。

なぜこのテストが必要か:
  - Pipeline が job_train / update_source_dataset / push_notebook ステップを
    正しく実行できることを保証する。
  - 既存の preprocess / train / inference ステップと共存することを確認する。
"""

from __future__ import annotations

import pytest
from omegaconf import DictConfig, OmegaConf

from src.usecase.pipeline.pipeline import PipelineUseCase


def _make_pipeline_cfg(steps: list[dict[str, object]]) -> DictConfig:
    return OmegaConf.create(
        {
            "usecase": "pipeline",
            "job_id": "test_job_pipeline",
            "steps": steps,
        }
    )


class TestPipelineJobTrainStep:
    def test_job_train_step_is_executed(self) -> None:
        """job_train ステップが実行されること。"""
        call_log: list[str] = []

        cfg = _make_pipeline_cfg([{"usecase": "job_train", "recipe": "lgbm"}])
        usecase = PipelineUseCase(
            run_preprocess=lambda c: call_log.append("preprocess"),
            run_train=lambda c: call_log.append("train"),
            run_inference=lambda c: call_log.append("inference"),
            run_job_train=lambda c: call_log.append("job_train"),
            run_update_source_dataset=lambda c: call_log.append("update_source_dataset"),
            run_push_notebook=lambda c: call_log.append("push_notebook"),
        )
        usecase.run(cfg)
        assert call_log == ["job_train"]

    def test_full_job_to_submission_pipeline(self) -> None:
        """full pipeline の全ステップが順に実行されること。"""
        call_log: list[str] = []

        cfg = _make_pipeline_cfg(
            [
                {"usecase": "preprocess", "recipe": "base"},
                {"usecase": "job_train", "recipe": "lgbm"},
                {"usecase": "inference", "recipe": "titanic_ensemble"},
                {"usecase": "update_source_dataset"},
                {"usecase": "push_notebook", "notebook": "titanic"},
            ]
        )
        usecase = PipelineUseCase(
            run_preprocess=lambda c: call_log.append("preprocess"),
            run_train=lambda c: call_log.append("train"),
            run_inference=lambda c: call_log.append("inference"),
            run_job_train=lambda c: call_log.append("job_train"),
            run_update_source_dataset=lambda c: call_log.append("update_source_dataset"),
            run_push_notebook=lambda c: call_log.append("push_notebook"),
        )
        usecase.run(cfg)
        assert call_log == [
            "preprocess",
            "job_train",
            "inference",
            "update_source_dataset",
            "push_notebook",
        ]

    def test_unknown_step_raises_value_error_with_new_runners(self) -> None:
        """未知の step.usecase で ValueError が送出されること（新 runner 追加後も）。"""
        cfg = _make_pipeline_cfg([{"usecase": "nonexistent_step"}])
        usecase = PipelineUseCase(
            run_preprocess=lambda c: None,
            run_train=lambda c: None,
            run_inference=lambda c: None,
            run_job_train=lambda c: None,
            run_update_source_dataset=lambda c: None,
            run_push_notebook=lambda c: None,
        )
        with pytest.raises(ValueError, match="nonexistent_step"):
            usecase.run(cfg)
