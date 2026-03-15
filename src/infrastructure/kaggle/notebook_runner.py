"""
NotebookPipelineRunner — Kaggle Notebook 上でパイプラインを実行するランナー。

conf/ から recipe config を読み込み、KaggleEnvironment でパスを差し替え、
OVERRIDES を適用してパイプラインを実行する。

Notebook セル3 から以下のように呼ぶだけでよい:

    runner = NotebookPipelineRunner(
        competition_slug=COMPETITION_SLUG,
        recipe=RECIPE,
        overrides=OVERRIDES,
    )
    submission_path = runner.run()
    print(f"[INFO] submission: {submission_path}")

設計上の注意:
- _run_preprocess / _run_train / _run_inference はテスト時に差し替え可能なよう
  インスタンス属性として保持する。
- Hydra struct モード制約を回避するため、OmegaConf.to_container() 経由で
  plain dict に変換してからパスを書き換え、OmegaConf.update() で OVERRIDES を適用する。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.infrastructure.kaggle.environment import KaggleEnvironment

logger = logging.getLogger(__name__)


class NotebookPipelineRunner:
    """Kaggle Notebook 上でパイプラインを実行するランナー。

    Args:
        competition_slug: Kaggle の competition slug（例: "titanic"）。
        recipe: conf/recipe/ 以下の yaml ファイル名（例: "base"）。
        conf_dir: conf/ ディレクトリのパス（デフォルト: "/kaggle/input/mlops-pipeline-src/conf"）。
        overrides: cfg に上書きする設定辞書。ドット記法のキーを使用する
                   （例: {"lgbm.num_leaves": 64}）。
    """

    def __init__(
        self,
        competition_slug: str,
        recipe: str,
        conf_dir: str = "/kaggle/input/mlops-pipeline-src/conf",
        overrides: dict[str, Any] | None = None,
    ) -> None:
        self._competition_slug = competition_slug
        self._recipe = recipe
        self._conf_dir = Path(conf_dir)
        self._overrides = overrides or {}

        # テスト時に差し替え可能なランナー関数
        self._run_preprocess = self._default_run_preprocess
        self._run_train = self._default_run_train
        self._run_inference = self._default_run_inference

    def run(self) -> Path:
        """パイプラインを実行し、submission.csv のパスを返す。

        処理フロー:
        1. conf_dir/recipe/{recipe}.yaml を OmegaConf.load で読み込む
        2. KaggleEnvironment でパスを解決する（input_root / output_root）
        3. recipe cfg の各 step のパス（PLACEHOLDER）を Kaggle パスに差し替える
        4. OVERRIDES を cfg に適用する
        5. preprocess → train → inference の順に実行する

        Returns:
            inference の返り値（submission.csv のパス）。

        Raises:
            FileNotFoundError: recipe yaml が見つからない場合。
            ValueError: inference step が見つからない場合。
        """
        recipe_path = self._conf_dir / "recipe" / f"{self._recipe}.yaml"
        if not recipe_path.exists():
            raise FileNotFoundError(
                f"recipe yaml が見つかりません: {recipe_path}\n"
                f"conf_dir={self._conf_dir}, recipe={self._recipe}"
            )

        # Hydra struct モード回避: OmegaConf.load は非 struct DictConfig を返すが
        # 明示的に to_container → create で plain dict DictConfig にする
        raw_cfg = OmegaConf.create(
            OmegaConf.to_container(OmegaConf.load(recipe_path), resolve=True)
        )

        # パス解決
        input_root = KaggleEnvironment.resolve_input_root(self._competition_slug)
        output_root = KaggleEnvironment.resolve_output_root()

        # OVERRIDES を適用（ドット記法サポート）
        for key, value in self._overrides.items():
            OmegaConf.update(raw_cfg, key, value, merge=True)

        # 各 step を順番に実行
        submission_path: Path | None = None
        steps_raw = OmegaConf.to_container(raw_cfg, resolve=True)
        assert isinstance(steps_raw, dict)
        steps = steps_raw.get("steps", [])
        for step in steps:
            assert isinstance(step, dict)
            step_usecase = str(step.get("usecase", ""))
            # step を plain dict DictConfig に変換してパスを差し替え
            patched: dict[str, Any] = self._patch_paths(step, input_root, output_root)
            step_cfg = DictConfig(patched)
            # OVERRIDES をマージ
            for key, value in self._overrides.items():
                OmegaConf.update(step_cfg, key, value, merge=True)

            logger.info("[NotebookPipelineRunner] Running step: %s", step_usecase)
            if step_usecase == "preprocess":
                self._run_preprocess(step_cfg)
            elif step_usecase == "train":
                self._run_train(step_cfg)
            elif step_usecase == "inference":
                result = self._run_inference(step_cfg)
                if result is not None:
                    submission_path = Path(str(result))

        if submission_path is None:
            raise ValueError(
                "pipeline に inference step がなかったため submission_path が未設定です。"
            )
        return submission_path

    def _patch_paths(
        self,
        step_dict: dict[str, Any],
        input_root: Path,
        output_root: Path,
    ) -> dict[str, Any]:
        """step_dict 内の PLACEHOLDER_* を Kaggle パスに差し替えて返す。

        preprocess step の inputs[].path と output_dir、
        train step の output_dir、
        inference step の test_path と output_dir を解決する。

        実際のパスは PLACEHOLDER_* 文字列ではなく、recipe yaml で設定されたパスを使う。
        Kaggle 環境では input_root / output_root で解決する。
        """
        step_usecase = step_dict.get("usecase", "")
        patched = dict(step_dict)

        if step_usecase == "preprocess":
            # inputs のパスを /kaggle/input/{competition_slug}/ 以下に差し替える
            inputs = step_dict.get("inputs", [])
            patched_inputs = []
            for inp in inputs:
                inp_dict = dict(inp)
                file_name = Path(str(inp_dict.get("path", ""))).name
                if not file_name or file_name.startswith("PLACEHOLDER"):
                    # ファイル名が不明な場合はデフォルトで train.csv / test.csv を使う
                    inp_id = str(inp_dict.get("id", ""))
                    if "test" in inp_id:
                        file_name = "test.csv"
                    else:
                        file_name = "train.csv"
                inp_dict["path"] = str(input_root / file_name)
                patched_inputs.append(inp_dict)
            patched["inputs"] = patched_inputs
            patched["output_dir"] = str(output_root / "processed")

        elif step_usecase == "train":
            patched["output_dir"] = str(output_root / "models")

        elif step_usecase == "inference":
            # test.parquet のパスを解決する
            # recipe yaml で明示されていない場合は汎用パスを使う
            original_test_path = str(step_dict.get("test_path", ""))
            if original_test_path and not original_test_path.startswith("PLACEHOLDER"):
                # recipe に明示的なパスがある場合はそのまま使う
                pass
            else:
                # デフォルト: processed/{competition_slug}_preprocess/latest/test_out/test.parquet
                preprocess_job_id = f"{self._competition_slug}_preprocess"
                patched["test_path"] = str(
                    output_root
                    / "processed"
                    / preprocess_job_id
                    / "latest"
                    / "test_out"
                    / "test.parquet"
                )
            patched["output_dir"] = str(output_root / "inference")

        return patched

    def _default_run_preprocess(self, cfg: DictConfig) -> None:
        """PreprocessUseCase をデフォルト DI で実行する（Notebook 本番用）。"""
        from src.infrastructure.executor.factory import ExecutorFactory
        from src.infrastructure.preprocessor.cv_splitter import CVSplitter
        from src.infrastructure.preprocessor.input_loader import InputLoader
        from src.infrastructure.preprocessor.visualizer import PipelineVisualizer
        from src.infrastructure.repository.git import GitRepositoryImpl
        from src.usecase.preprocessing.preprocess import PreprocessUseCase

        git_repo = GitRepositoryImpl()
        executor, _ = ExecutorFactory.build_with_fallback("local")
        PreprocessUseCase(
            cfg,
            executor=executor,
            git_repo=git_repo,
            input_loader=InputLoader(),
            cv_splitter=CVSplitter(),
            visualizer=PipelineVisualizer(),
        ).execute()

    def _default_run_train(self, cfg: DictConfig) -> None:
        """TrainUseCase をデフォルト DI で実行する（Notebook 本番用）。"""
        from src.infrastructure.repository.git import GitRepositoryImpl
        from src.infrastructure.trainer.lgbm_trainer import LightGBMTrainer
        from src.usecase.training.train import TrainUseCase

        git_repo = GitRepositoryImpl()
        TrainUseCase(cfg, trainer=LightGBMTrainer(cfg), git_repo=git_repo).execute()

    def _default_run_inference(self, cfg: DictConfig) -> Path | None:
        """InferenceUseCase をデフォルト DI で実行する（Notebook 本番用）。"""
        from src.infrastructure.inference.lgbm_inferencer import LightGBMInferencer
        from src.infrastructure.repository.git import GitRepositoryImpl
        from src.usecase.inference.inference import InferenceUseCase

        git_repo = GitRepositoryImpl()
        return InferenceUseCase(inferencer=LightGBMInferencer(), git_repo=git_repo).run(cfg)
