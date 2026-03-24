"""
NotebookPipelineRunner — Kaggle Notebook 上でパイプラインを実行するランナー。

conf/competition/{slug}/pipeline/{recipe}.yaml を読み込み、各 step の設定を
conf/competition/{slug}/{usecase_dir}/{step_recipe}.yaml からロードして
KaggleEnvironment でパスを差し替え、パイプラインを実行する。

Notebook セル3 から以下のように呼ぶだけでよい:

    runner = NotebookPipelineRunner(
        competition_slug=COMPETITION_SLUG,
        recipe=RECIPE,
        overrides=OVERRIDES,
    )
    submission_path = runner.run()
    print(f"[INFO] submission: {submission_path}")

設計上の注意:
- pipeline yaml の各 step は usecase と recipe のみ持ち、フル設定は
  conf/competition/{slug}/{usecase_dir}/{step_recipe}.yaml からロードする。
- _run_preprocess / _run_train / _run_inference はテスト時に差し替え可能なよう
  インスタンス属性として保持する。
- ${competition.name} などの OmegaConf 補間を解決するため、各 step yaml は
  {competition: {name: slug}} をベース設定としてマージしてから resolve する。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.infrastructure.kaggle.environment import KaggleEnvironment

logger = logging.getLogger(__name__)

# usecase 名 → conf/ のサブディレクトリ名 のマッピング
_USECASE_TO_CONF_SUBDIR: dict[str, str] = {
    "preprocess": "preprocess",
    "train": "training",
    "inference": "inference",
}


class NotebookPipelineRunner:
    """Kaggle Notebook 上でパイプラインを実行するランナー。

    Args:
        competition_slug: competition slug（例: "my-competition"）。
        recipe: conf/competition/{slug}/pipeline/ 以下の yaml ファイル名
                （例: "all_after_download"）。
        conf_dir: conf/ ディレクトリのパス（デフォルト: "/kaggle/input/mlops-pipeline-src/conf"）。
        overrides: cfg に上書きする設定辞書。ドット記法のキーを使用する
                   （例: {"lgbm.num_leaves": 64}）。
    """

    def __init__(
        self,
        competition_slug: str,
        recipe: str,
        conf_dir: str = "/kaggle/working/conf",
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
        1. conf/competition/{slug}/pipeline/{recipe}.yaml を読み込む
        2. 各 step の設定を conf/competition/{slug}/{usecase_dir}/{step_recipe}.yaml からロードする
        3. KaggleEnvironment でパスを解決する（input_root / output_root）
        4. 各 step のパスを Kaggle パスに差し替える
        5. OVERRIDES を cfg に適用する
        6. preprocess → train → inference の順に実行する

        Returns:
            inference の返り値（submission.csv のパス）。

        Raises:
            FileNotFoundError: pipeline yaml または step yaml が見つからない場合。
            ValueError: inference step が見つからない場合。
        """
        pipeline_path = (
            self._conf_dir
            / "competition"
            / self._competition_slug
            / "pipeline"
            / f"{self._recipe}.yaml"
        )
        if not pipeline_path.exists():
            raise FileNotFoundError(
                f"pipeline yaml が見つかりません: {pipeline_path}\n"
                f"conf_dir={self._conf_dir}, competition={self._competition_slug}, "
                f"recipe={self._recipe}"
            )

        pipeline_raw = OmegaConf.to_container(OmegaConf.load(pipeline_path), resolve=False)
        assert isinstance(pipeline_raw, dict)
        steps = pipeline_raw.get("steps", [])

        # Kaggle パス解決
        input_root = KaggleEnvironment.resolve_input_root(self._competition_slug)
        output_root = KaggleEnvironment.resolve_output_root()

        submission_path: Path | None = None
        for step in steps:
            assert isinstance(step, dict)
            step_usecase = str(step.get("usecase", ""))
            step_recipe = str(step.get("recipe", ""))

            # step-specific config をロード
            conf_subdir = _USECASE_TO_CONF_SUBDIR.get(step_usecase, step_usecase)
            step_cfg_path = (
                self._conf_dir
                / "competition"
                / self._competition_slug
                / conf_subdir
                / f"{step_recipe}.yaml"
            )
            if not step_cfg_path.exists():
                raise FileNotFoundError(
                    f"step config が見つかりません: {step_cfg_path}\n"
                    f"usecase={step_usecase}, recipe={step_recipe}"
                )

            # ${competition.name} などの補間を解決するためベース設定をマージしてから resolve
            base = OmegaConf.create({"competition": {"name": self._competition_slug}})
            merged = OmegaConf.merge(base, OmegaConf.load(step_cfg_path))
            step_raw_container = OmegaConf.to_container(merged, resolve=True)
            assert isinstance(step_raw_container, dict)
            step_raw: dict[str, Any] = {str(k): v for k, v in step_raw_container.items()}

            # パスを Kaggle 環境用に差し替え
            patched = self._patch_paths(step_raw, input_root, output_root)
            step_cfg = DictConfig(patched)

            # OVERRIDES を適用（ドット記法サポート）
            for key, value in self._overrides.items():
                OmegaConf.update(step_cfg, key, value, merge=True)

            logger.info(
                "[NotebookPipelineRunner] Running step: %s (recipe: %s)", step_usecase, step_recipe
            )
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
        """step_dict の各パスを Kaggle 環境パスに上書きして返す。

        ローカルの data/ や models/ といった相対パスを Kaggle の絶対パスに変換する。

        - preprocess:
            inputs[].path → input_root/{filename}
            output_dir    → output_root/processed
        - train:
            preprocess_output_dir → output_root/processed/{slug}_preprocess/latest/train_out
            output_dir            → output_root/models/{slug}
        - inference:
            test_path  → output_root/processed/{slug}_preprocess/latest/test_out/test.parquet
            models[*]  → output_root/models/{slug}/{job_id}/latest
            output_dir → output_root/inference/{slug}
        """
        step_usecase = step_dict.get("usecase", "")
        patched = dict(step_dict)
        preprocess_job_id = f"{self._competition_slug}_preprocess"

        if step_usecase == "preprocess":
            inputs = step_dict.get("inputs", [])
            patched_inputs = []
            for inp in inputs:
                inp_dict = dict(inp)
                file_name = Path(str(inp_dict.get("path", ""))).name
                if not file_name or file_name.startswith("PLACEHOLDER"):
                    inp_id = str(inp_dict.get("id", ""))
                    file_name = "test.csv" if "test" in inp_id else "train.csv"
                inp_dict["path"] = str(input_root / file_name)
                patched_inputs.append(inp_dict)
            patched["inputs"] = patched_inputs
            patched["output_dir"] = str(output_root / "processed")

        elif step_usecase == "train":
            patched["preprocess_output_dir"] = str(
                output_root / "processed" / preprocess_job_id / "latest" / "train_out"
            )
            patched["output_dir"] = str(output_root / "models" / self._competition_slug)

        elif step_usecase == "inference":
            patched["test_path"] = str(
                output_root
                / "processed"
                / preprocess_job_id
                / "latest"
                / "test_out"
                / "test.parquet"
            )
            # models パスは "models/{slug}/{job_id}/latest" 形式 → output_root 以下に変換
            models = step_dict.get("models", [])
            patched["models"] = [str(output_root / m) for m in models]
            patched["output_dir"] = str(output_root / "inference" / self._competition_slug)

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
