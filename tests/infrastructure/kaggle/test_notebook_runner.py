"""
NotebookPipelineRunner のテスト。

なぜこのテストが必要か:
  - Kaggle Notebook 上でパイプライン（preprocess/train/inference）を順番に実行する
    NotebookPipelineRunner が正しく動作することを保証する。
  - OVERRIDES が cfg に反映されること、KaggleEnvironment でパスが解決されること、
    submission.csv のパスが返されることを確認する。
  - UseCase は MagicMock で差し替えて、Kaggle 実環境に依存しないテストを書く。

fixture:
  - monkeypatch: KAGGLE_KERNEL_RUN_TYPE=Interactive を設定して Kaggle 環境を擬似再現
  - tmp_path: 一時的な conf_dir を作成してレシピ yaml を置く
  - MagicMock: PreprocessUseCase / TrainUseCase / InferenceUseCase を差し替える
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.infrastructure.kaggle.notebook_runner import NotebookPipelineRunner


def _write_recipe_yaml(conf_dir: Path, recipe: str) -> Path:
    """テスト用の recipe yaml を conf_dir/recipe/{recipe}.yaml に作成する。"""
    recipe_dir = conf_dir / "recipe"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = recipe_dir / f"{recipe}.yaml"
    recipe_path.write_text(
        "steps:\n"
        "  - usecase: preprocess\n"
        "    job_id: titanic_preprocess\n"
        "    inputs:\n"
        "      - id: raw_train\n"
        "        path: PLACEHOLDER_TRAIN\n"
        "        format: csv\n"
        "      - id: raw_test\n"
        "        path: PLACEHOLDER_TEST\n"
        "        format: csv\n"
        "    output_dir: PLACEHOLDER_OUTPUT\n"
        "  - usecase: train\n"
        "    job_id: titanic_lgbm\n"
        "    output_dir: PLACEHOLDER_MODELS\n"
        "  - usecase: inference\n"
        "    job_id: titanic_inference\n"
        "    test_path: PLACEHOLDER_TEST_PATH\n"
        "    output_dir: PLACEHOLDER_INFERENCE\n"
    )
    return recipe_path


class TestNotebookPipelineRunnerCallsPipelineSteps:
    """preprocess/train/inference が順番に呼ばれることを検証する。"""

    def test_run_calls_pipeline_steps(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """run() が preprocess → train → inference の順に UseCase を呼ぶこと。"""
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
        conf_dir = tmp_path / "conf"
        _write_recipe_yaml(conf_dir, "base")

        call_order: list[str] = []

        def _append_preprocess(cfg: object) -> None:
            call_order.append("preprocess")

        def _append_train(cfg: object) -> None:
            call_order.append("train")

        def _append_inference(cfg: object) -> Path:
            call_order.append("inference")
            return Path("/kaggle/working/inference/submission.csv")

        mock_preprocess = MagicMock(side_effect=_append_preprocess)
        mock_train = MagicMock(side_effect=_append_train)
        mock_inference = MagicMock(side_effect=_append_inference)

        runner = NotebookPipelineRunner(
            competition_slug="titanic",
            recipe="base",
            conf_dir=str(conf_dir),
        )
        runner._run_preprocess = mock_preprocess
        runner._run_train = mock_train
        runner._run_inference = mock_inference

        runner.run()

        assert call_order == ["preprocess", "train", "inference"], (
            f"UseCase の呼び出し順が正しくない: {call_order}"
        )


class TestNotebookPipelineRunnerKagglePaths:
    """input_root が /kaggle/input/{slug} に解決されることを検証する。"""

    def test_kaggle_paths_are_resolved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Kaggle 環境では input_root が /kaggle/input/titanic になること。"""
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
        conf_dir = tmp_path / "conf"
        _write_recipe_yaml(conf_dir, "base")

        from omegaconf import DictConfig

        captured_cfgs: list[DictConfig] = []

        def capture_preprocess(cfg: DictConfig) -> None:
            captured_cfgs.append(cfg)

        mock_train = MagicMock()
        mock_inference = MagicMock(return_value=Path("/kaggle/working/inference/submission.csv"))

        runner = NotebookPipelineRunner(
            competition_slug="titanic",
            recipe="base",
            conf_dir=str(conf_dir),
        )
        runner._run_preprocess = capture_preprocess
        runner._run_train = mock_train
        runner._run_inference = mock_inference

        runner.run()

        assert len(captured_cfgs) == 1, "preprocess が1回呼ばれていない"
        preprocess_cfg = captured_cfgs[0]
        # inputs[0].path が /kaggle/input/titanic 以下に解決されていることを確認
        input_path = str(preprocess_cfg.inputs[0].path)
        assert "/kaggle/input/titanic" in input_path, (
            f"input_root が /kaggle/input/titanic に解決されていない: {input_path}"
        )


class TestNotebookPipelineRunnerOverrides:
    """OVERRIDES の値が cfg に反映されることを検証する。"""

    def test_overrides_are_applied(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """OVERRIDES の値が各 step の cfg に反映されること。"""
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
        conf_dir = tmp_path / "conf"
        _write_recipe_yaml(conf_dir, "base")

        from omegaconf import DictConfig

        captured_cfgs: list[DictConfig] = []

        def capture_train(cfg: DictConfig) -> None:
            captured_cfgs.append(cfg)

        mock_preprocess = MagicMock()
        mock_inference = MagicMock(return_value=Path("/kaggle/working/inference/submission.csv"))

        runner = NotebookPipelineRunner(
            competition_slug="titanic",
            recipe="base",
            conf_dir=str(conf_dir),
            overrides={"lgbm.num_leaves": 64},
        )
        runner._run_preprocess = mock_preprocess
        runner._run_train = capture_train
        runner._run_inference = mock_inference

        runner.run()

        assert len(captured_cfgs) == 1, "train が1回呼ばれていない"
        train_cfg = captured_cfgs[0]
        # OVERRIDES の値が cfg に反映されていることを確認
        assert train_cfg.lgbm.num_leaves == 64, (
            f"OVERRIDES が反映されていない: lgbm.num_leaves = {train_cfg.get('lgbm')}"
        )


class TestNotebookPipelineRunnerReturnsSubmissionPath:
    """run() が submission.csv のパスを返すことを検証する。"""

    def test_run_returns_submission_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """run() が inference の返り値（submission.csv のパス）を返すこと。"""
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
        conf_dir = tmp_path / "conf"
        _write_recipe_yaml(conf_dir, "base")

        expected_path = Path("/kaggle/working/inference/titanic_inference/latest/submission.csv")

        mock_preprocess = MagicMock()
        mock_train = MagicMock()
        mock_inference = MagicMock(return_value=expected_path)

        runner = NotebookPipelineRunner(
            competition_slug="titanic",
            recipe="base",
            conf_dir=str(conf_dir),
        )
        runner._run_preprocess = mock_preprocess
        runner._run_train = mock_train
        runner._run_inference = mock_inference

        result = runner.run()

        assert result == expected_path, f"run() の返り値が期待値と異なる: {result}"
