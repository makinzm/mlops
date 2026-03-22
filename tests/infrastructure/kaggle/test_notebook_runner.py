"""
NotebookPipelineRunner のテスト。

なぜこのテストが必要か:
  - Kaggle Notebook 上でパイプライン（preprocess/train/inference）を順番に実行する
    NotebookPipelineRunner が正しく動作することを保証する。
  - pipeline yaml は conf/competition/{slug}/pipeline/{recipe}.yaml にあり、
    各 step の設定は conf/competition/{slug}/{usecase_dir}/{step_recipe}.yaml に分離されている。
  - OVERRIDES が cfg に反映されること、KaggleEnvironment でパスが解決されること、
    submission.csv のパスが返されることを確認する。
  - UseCase は MagicMock で差し替えて、Kaggle 実環境に依存しないテストを書く。

fixture:
  - monkeypatch: KAGGLE_KERNEL_RUN_TYPE=Interactive を設定して Kaggle 環境を擬似再現
  - tmp_path: 一時的な conf_dir を作成してレシピ yaml を置く
  - MagicMock: _run_preprocess / _run_train / _run_inference を差し替える
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.infrastructure.kaggle.notebook_runner import NotebookPipelineRunner


def _write_test_conf(conf_dir: Path, slug: str = "titanic", recipe: str = "base") -> None:
    """テスト用の conf/ 構造を作成する。

    conf/
      competition/
        {slug}/
          pipeline/
            {recipe}.yaml  ← steps のみ記述
          preprocess/
            base.yaml
          training/
            lgbm.yaml
          inference/
            titanic_ensemble.yaml
    """
    comp_dir = conf_dir / "competition" / slug

    # pipeline yaml（steps: usecase + recipe のみ）
    pipeline_dir = comp_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    (pipeline_dir / f"{recipe}.yaml").write_text(
        "steps:\n"
        "  - usecase: preprocess\n"
        "    recipe: base\n"
        "  - usecase: train\n"
        "    recipe: lgbm\n"
        "  - usecase: inference\n"
        "    recipe: titanic_ensemble\n"
    )

    # preprocess step yaml
    preprocess_dir = comp_dir / "preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    (preprocess_dir / "base.yaml").write_text(
        "usecase: preprocess\n"
        "job_id: titanic_preprocess\n"
        "inputs:\n"
        "  - id: raw_train\n"
        "    path: data/raw/train.csv\n"
        "    format: csv\n"
        "  - id: raw_test\n"
        "    path: data/raw/test.csv\n"
        "    format: csv\n"
        "output_dir: data/processed\n"
    )

    # training step yaml
    training_dir = comp_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    (training_dir / "lgbm.yaml").write_text(
        "usecase: train\n"
        "job_id: titanic_lgbm\n"
        "trainer:\n"
        "  type: lgbm\n"
        "preprocess_output_dir: data/processed/titanic_preprocess/latest/train_out\n"
        "output_dir: models/titanic\n"
        "lgbm:\n"
        "  num_leaves: 31\n"
    )

    # inference step yaml
    inference_dir = comp_dir / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    (inference_dir / "titanic_ensemble.yaml").write_text(
        "usecase: inference\n"
        "job_id: titanic_inference\n"
        "test_path: data/processed/titanic_preprocess/latest/test_out/test.parquet\n"
        "models:\n"
        "  - models/titanic/titanic_lgbm/latest\n"
        "output_dir: data/inference/titanic\n"
    )


class TestNotebookPipelineRunnerCallsPipelineSteps:
    """preprocess/train/inference が順番に呼ばれることを検証する。"""

    def test_run_calls_pipeline_steps(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """run() が preprocess → train → inference の順に UseCase を呼ぶこと。"""
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
        conf_dir = tmp_path / "conf"
        _write_test_conf(conf_dir, slug="titanic", recipe="base")

        call_order: list[str] = []

        def _append_preprocess(cfg: object) -> None:
            call_order.append("preprocess")

        def _append_train(cfg: object) -> None:
            call_order.append("train")

        def _append_inference(cfg: object) -> Path:
            call_order.append("inference")
            return Path("/kaggle/working/inference/submission.csv")

        runner = NotebookPipelineRunner(
            competition_slug="titanic",
            recipe="base",
            conf_dir=str(conf_dir),
        )
        runner._run_preprocess = MagicMock(side_effect=_append_preprocess)
        runner._run_train = MagicMock(side_effect=_append_train)
        runner._run_inference = MagicMock(side_effect=_append_inference)

        runner.run()

        assert call_order == ["preprocess", "train", "inference"], (
            f"UseCase の呼び出し順が正しくない: {call_order}"
        )


class TestNotebookPipelineRunnerKagglePaths:
    """Kaggle 環境でパスが正しく解決されることを検証する。"""

    def test_preprocess_input_path_resolved_to_kaggle_input(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """preprocess の inputs[0].path が /kaggle/input/competitions/titanic
        以下に解決されること。"""
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
        conf_dir = tmp_path / "conf"
        _write_test_conf(conf_dir)

        from omegaconf import DictConfig

        captured_cfgs: list[DictConfig] = []

        def capture_preprocess(cfg: DictConfig) -> None:
            captured_cfgs.append(cfg)

        runner = NotebookPipelineRunner(
            competition_slug="titanic",
            recipe="base",
            conf_dir=str(conf_dir),
        )
        runner._run_preprocess = capture_preprocess
        runner._run_train = MagicMock()
        runner._run_inference = MagicMock(
            return_value=Path("/kaggle/working/inference/submission.csv")
        )

        runner.run()

        assert len(captured_cfgs) == 1, "preprocess が1回呼ばれていない"
        input_path = str(captured_cfgs[0].inputs[0].path)
        assert "/kaggle/input/competitions/titanic" in input_path, (
            f"input_root が /kaggle/input/competitions/titanic に解決されていない: {input_path}"
        )

    def test_train_preprocess_output_dir_resolved_to_kaggle_working(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """train の preprocess_output_dir が /kaggle/working/processed 以下に解決されること。"""
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
        conf_dir = tmp_path / "conf"
        _write_test_conf(conf_dir)

        from omegaconf import DictConfig

        captured_cfgs: list[DictConfig] = []

        def capture_train(cfg: DictConfig) -> None:
            captured_cfgs.append(cfg)

        runner = NotebookPipelineRunner(
            competition_slug="titanic",
            recipe="base",
            conf_dir=str(conf_dir),
        )
        runner._run_preprocess = MagicMock()
        runner._run_train = capture_train
        runner._run_inference = MagicMock(
            return_value=Path("/kaggle/working/inference/submission.csv")
        )

        runner.run()

        assert len(captured_cfgs) == 1, "train が1回呼ばれていない"
        preprocess_output = str(captured_cfgs[0].preprocess_output_dir)
        assert "/kaggle/working/processed" in preprocess_output, (
            "preprocess_output_dir が /kaggle/working/processed に解決されていない: "
            f"{preprocess_output}"
        )

    def test_inference_test_path_resolved_to_kaggle_working(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """inference の test_path が /kaggle/working/processed 以下に解決されること。"""
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
        conf_dir = tmp_path / "conf"
        _write_test_conf(conf_dir)

        from omegaconf import DictConfig

        captured_cfgs: list[DictConfig] = []

        def capture_inference(cfg: DictConfig) -> Path:
            captured_cfgs.append(cfg)
            return Path("/kaggle/working/inference/submission.csv")

        runner = NotebookPipelineRunner(
            competition_slug="titanic",
            recipe="base",
            conf_dir=str(conf_dir),
        )
        runner._run_preprocess = MagicMock()
        runner._run_train = MagicMock()
        runner._run_inference = capture_inference

        runner.run()

        assert len(captured_cfgs) == 1, "inference が1回呼ばれていない"
        test_path = str(captured_cfgs[0].test_path)
        assert "/kaggle/working/processed" in test_path, (
            f"test_path が /kaggle/working/processed に解決されていない: {test_path}"
        )


class TestNotebookPipelineRunnerOverrides:
    """OVERRIDES の値が cfg に反映されることを検証する。"""

    def test_overrides_are_applied_to_train_cfg(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """OVERRIDES の値が train step の cfg に反映されること。"""
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
        conf_dir = tmp_path / "conf"
        _write_test_conf(conf_dir)

        from omegaconf import DictConfig

        captured_cfgs: list[DictConfig] = []

        def capture_train(cfg: DictConfig) -> None:
            captured_cfgs.append(cfg)

        runner = NotebookPipelineRunner(
            competition_slug="titanic",
            recipe="base",
            conf_dir=str(conf_dir),
            overrides={"lgbm.num_leaves": 64},
        )
        runner._run_preprocess = MagicMock()
        runner._run_train = capture_train
        runner._run_inference = MagicMock(
            return_value=Path("/kaggle/working/inference/submission.csv")
        )

        runner.run()

        assert len(captured_cfgs) == 1, "train が1回呼ばれていない"
        assert captured_cfgs[0].lgbm.num_leaves == 64, (
            f"OVERRIDES が反映されていない: lgbm.num_leaves = {captured_cfgs[0].get('lgbm')}"
        )


class TestNotebookPipelineRunnerReturnsSubmissionPath:
    """run() が submission.csv のパスを返すことを検証する。"""

    def test_run_returns_submission_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """run() が inference の返り値（submission.csv のパス）を返すこと。"""
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
        conf_dir = tmp_path / "conf"
        _write_test_conf(conf_dir)

        expected_path = Path("/kaggle/working/inference/titanic_inference/latest/submission.csv")

        runner = NotebookPipelineRunner(
            competition_slug="titanic",
            recipe="base",
            conf_dir=str(conf_dir),
        )
        runner._run_preprocess = MagicMock()
        runner._run_train = MagicMock()
        runner._run_inference = MagicMock(return_value=expected_path)

        result = runner.run()

        assert result == expected_path, f"run() の返り値が期待値と異なる: {result}"


class TestNotebookPipelineRunnerMissingFiles:
    """yaml ファイルが見つからない場合に FileNotFoundError が上がることを検証する。"""

    def test_missing_pipeline_yaml_raises_file_not_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """pipeline yaml がない場合に FileNotFoundError が上がること。"""
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()

        runner = NotebookPipelineRunner(
            competition_slug="titanic",
            recipe="nonexistent",
            conf_dir=str(conf_dir),
        )
        with pytest.raises(FileNotFoundError, match="pipeline yaml"):
            runner.run()
