"""
InferenceUseCase — 推論のユースケース。

Inferencer Protocol と GitRepository Protocol を受け取り、
- test_path からテストデータを読み込む
- Inferencer.predict_folds() で各 fold の予測値を取得
- EnsembleStrategy で集約して最終予測を生成
- submission.csv / metainfo.yaml / README.md / .gitignore を output_dir に生成
を担う。

出力ディレクトリ構造:
  {output_dir}/{job_id}/
    ├── .gitignore
    └── {YYYYMMDDTHHMMSS}/
        ├── submission.csv     # test_path が有効な場合のみ生成
        ├── metainfo.yaml
        └── README.md

test_path が null またはファイルが存在しない場合は submission.csv の生成をスキップし、
metainfo.yaml と README.md のみ生成する（test データなしコンペでも実行可能）。

全パスは Hydra Config で管理する。
per-directory .gitignore は {output_dir}/{job_id}/ に動的生成する。
（ルート .gitignore への追記禁止）
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from omegaconf import DictConfig

from src.domain.repository.git import GitRepository
from src.usecase._utils import build_tree_lines, resolve_latest_dir
from src.usecase.inference.ensemble_strategies import (
    EnsembleStrategy,
    MeanStrategy,
    RankAverageStrategy,
    WeightedMeanStrategy,
)
from src.usecase.inference.inferencer import Inferencer

logger = logging.getLogger(__name__)

_INFERENCE_OUTPUT_GITIGNORE = """\
*
!.gitignore
!*.yaml
!*.md
!*/
"""


def _resolve_strategy(cfg: DictConfig) -> EnsembleStrategy:
    """cfg.ensemble に応じた EnsembleStrategy を返す。"""
    ensemble = str(cfg.get("ensemble", "mean"))
    if ensemble == "mean":
        return MeanStrategy()
    elif ensemble == "weighted_mean":
        weights: list[float] = list(cfg.get("weights", [1.0]))
        return WeightedMeanStrategy(weights=weights)
    elif ensemble == "rank_average":
        return RankAverageStrategy()
    else:
        raise ValueError(f"unknown ensemble strategy: {ensemble!r}")


class InferenceUseCase:
    """Inferencer と GitRepository を受け取り推論を実行する。"""

    def __init__(
        self,
        inferencer: Inferencer,
        git_repo: GitRepository,
    ) -> None:
        self._inferencer = inferencer
        self._git_repo = git_repo

    def run(self, cfg: DictConfig) -> Path | None:
        """推論を実行し、submission.csv を生成する。

        Args:
            cfg: Hydra DictConfig（test_path, feature_cols, models, output_dir 等を含む）

        Returns:
            submission.csv のパス（{output_dir}/{job_id}/{timestamp}/submission.csv）。
            test_path が null またはファイル不在の場合は None。
        """
        job_id: str = str(cfg.get("job_id", "inference"))
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

        # {output_dir}/{job_id}/ に .gitignore を配置（モデル出力と同じ方式）
        job_output_dir = Path(str(cfg.output_dir)) / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)

        gitignore_path = job_output_dir / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text(_INFERENCE_OUTPUT_GITIGNORE)

        # タイムスタンプ付きディレクトリに全ファイルを配置
        ts_dir = job_output_dir / timestamp
        ts_dir.mkdir(parents=True, exist_ok=True)

        # test_path の有効性チェック
        test_path_raw = cfg.get("test_path")
        has_test_data = self._resolve_test_path(test_path_raw)

        feature_cols: list[str] = list(cfg.feature_cols)
        passenger_id_col: str = str(cfg.get("passenger_id_col", "PassengerId"))
        submission_path: Path | None = None
        predictions: list[np.ndarray] = []
        n_test = 0

        if has_test_data is not None:
            test_df = pl.read_parquet(str(has_test_data))
            n_test = len(test_df)

            # models ディレクトリのリストから予測
            strategy = _resolve_strategy(cfg)

            models_cfg = cfg.get("models", [])
            if not models_cfg:
                raise ValueError(
                    "cfg.models が空です。推論対象のモデルディレクトリを指定してください。"
                )

            for model_path_str in models_cfg:
                # "latest" を含むパスを最新タイムスタンプに解決
                model_dir = resolve_latest_dir(str(model_path_str))
                # モデルパスが fold_N/model.lgbm の場合は親ディレクトリを渡す
                if model_dir.suffix == ".lgbm":
                    model_dir = model_dir.parent.parent
                pred = self._inferencer.predict_folds(model_dir, test_df, feature_cols)
                predictions.append(pred)

            final_pred = strategy.aggregate(predictions)

            # submission.csv 生成（Kaggle フォーマット）
            passenger_ids = test_df[passenger_id_col].to_list()
            submission = pl.DataFrame(
                {
                    passenger_id_col: passenger_ids,
                    "Survived": final_pred.tolist(),
                }
            )
            submission_path = ts_dir / "submission.csv"
            submission.write_csv(str(submission_path))

        # metainfo.yaml に commit_hash / timestamp を記録
        commit_hash = self._git_repo.get_commit_hash()
        metainfo: dict[str, Any] = {
            "job_id": job_id,
            "timestamp": timestamp,
            "commit_hash": commit_hash,
            "ensemble": str(cfg.get("ensemble", "mean")),
            "n_models": len(predictions),
            "n_test": n_test,
        }
        metainfo_path = ts_dir / "metainfo.yaml"
        metainfo_path.write_text(yaml.dump(metainfo, allow_unicode=True, sort_keys=False))

        # README.md 生成
        self._write_readme(ts_dir, metainfo)

        return submission_path

    @staticmethod
    def _resolve_test_path(test_path_raw: object) -> Path | None:
        """test_path を解決して Path を返す。null またはファイル不在の場合は None を返す。"""
        if test_path_raw is None:
            logger.warning("test_path が null です。submission.csv の生成をスキップします。")
            return None
        resolved = resolve_latest_dir(str(test_path_raw))
        if not resolved.exists():
            logger.warning(
                "test_path %s が存在しません。submission.csv の生成をスキップします。",
                resolved,
            )
            return None
        return resolved

    @staticmethod
    def _write_readme(ts_dir: Path, metainfo: dict[str, Any]) -> None:
        lines: list[str] = [
            f"# Inference Output — `{metainfo['job_id']}`",
            "",
            f"- timestamp: `{metainfo['timestamp']}`",
            f"- commit: `{metainfo['commit_hash']}`",
            f"- ensemble: {metainfo['ensemble']}",
            f"- n_models: {metainfo['n_models']}",
            f"- n_test: {metainfo['n_test']}",
            "",
            "## Output Files",
            "",
            "```",
            ts_dir.name + "/",
        ]
        lines += build_tree_lines(ts_dir)
        lines.append("```")
        (ts_dir / "README.md").write_text("\n".join(lines) + "\n")
