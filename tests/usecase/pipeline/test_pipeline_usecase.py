"""
PipelineUseCase の単体テスト。

なぜこのテストが必要か:
  - PipelineUseCase は cfg.steps を順番に読み込み、
    step.usecase に応じて _run_preprocess / _run_train / _run_inference を順次呼ぶ。
  - 「順序通りに実行される」「1ステップが失敗したら後続が止まる」ことを
    テストで保証しないと、誤った順序での実行に気づけない。
  - 各 runner 関数を DI することで、インフラ依存なしに UseCase ロジックをテストできる。
  - 失敗時の早期終了（fail-fast）が実装されていることを確認する。
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest
from omegaconf import DictConfig, OmegaConf

from src.usecase.pipeline.pipeline import PipelineUseCase


# ──────────────────────────────────────────────────────────────
# ヘルパー
# ──────────────────────────────────────────────────────────────


def _make_pipeline_cfg(steps: list[dict]) -> DictConfig:
    """PipelineUseCase 用の DictConfig を生成する。"""
    return OmegaConf.create(
        {
            "usecase": "pipeline",
            "job_id": "test_pipeline",
            "steps": steps,
        }
    )


class TestPipelineUseCaseRun:
    def test_run_executes_steps_in_order(self) -> None:
        """steps が定義された順番に実行されること。"""
        call_order: list[str] = []

        def mock_preprocess(cfg: DictConfig) -> None:
            call_order.append("preprocess")

        def mock_train(cfg: DictConfig) -> None:
            call_order.append("train")

        def mock_inference(cfg: DictConfig) -> None:
            call_order.append("inference")

        cfg = _make_pipeline_cfg(
            [
                {"usecase": "preprocess", "recipe": "base"},
                {"usecase": "train", "recipe": "lgbm"},
                {"usecase": "inference", "recipe": "titanic_ensemble"},
            ]
        )
        usecase = PipelineUseCase(
            run_preprocess=mock_preprocess,
            run_train=mock_train,
            run_inference=mock_inference,
        )
        usecase.run(cfg)

        assert call_order == ["preprocess", "train", "inference"], (
            f"実行順序が期待と異なる: {call_order}"
        )

    def test_run_calls_each_runner_with_merged_cfg(self) -> None:
        """各 runner は step の設定がマージされた DictConfig を受け取ること。"""
        received_cfgs: list[DictConfig] = []

        def capture_cfg(cfg: DictConfig) -> None:
            received_cfgs.append(cfg)

        cfg = _make_pipeline_cfg(
            [
                {"usecase": "preprocess", "recipe": "base"},
            ]
        )
        usecase = PipelineUseCase(
            run_preprocess=capture_cfg,
            run_train=lambda c: None,
            run_inference=lambda c: None,
        )
        usecase.run(cfg)

        assert len(received_cfgs) == 1
        # step の recipe が cfg にマージされていること
        assert received_cfgs[0].get("recipe") == "base"

    def test_run_stops_on_failure(self) -> None:
        """1ステップが失敗したら後続ステップが実行されないこと（fail-fast）。"""
        executed: list[str] = []

        def fail_preprocess(cfg: DictConfig) -> None:
            executed.append("preprocess")
            raise RuntimeError("preprocess failed")

        def mock_train(cfg: DictConfig) -> None:
            executed.append("train")  # 呼ばれてはいけない

        cfg = _make_pipeline_cfg(
            [
                {"usecase": "preprocess", "recipe": "base"},
                {"usecase": "train", "recipe": "lgbm"},
            ]
        )
        usecase = PipelineUseCase(
            run_preprocess=fail_preprocess,
            run_train=mock_train,
            run_inference=lambda c: None,
        )
        with pytest.raises(RuntimeError, match="preprocess failed"):
            usecase.run(cfg)

        assert "train" not in executed, "fail-fast が機能していない"

    def test_run_raises_for_unknown_usecase(self) -> None:
        """step.usecase が未知の値のとき ValueError を送出すること。"""
        cfg = _make_pipeline_cfg(
            [
                {"usecase": "unknown_step", "recipe": "something"},
            ]
        )
        usecase = PipelineUseCase(
            run_preprocess=lambda c: None,
            run_train=lambda c: None,
            run_inference=lambda c: None,
        )
        with pytest.raises(ValueError, match="unknown_step"):
            usecase.run(cfg)

    def test_run_single_step(self) -> None:
        """ステップが 1 つだけでも正常に動作すること。"""
        executed: list[str] = []

        def mock_inference(cfg: DictConfig) -> None:
            executed.append("inference")

        cfg = _make_pipeline_cfg(
            [
                {"usecase": "inference", "recipe": "titanic_ensemble"},
            ]
        )
        usecase = PipelineUseCase(
            run_preprocess=lambda c: None,
            run_train=lambda c: None,
            run_inference=mock_inference,
        )
        usecase.run(cfg)

        assert executed == ["inference"]
