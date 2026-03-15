"""
InferenceUseCase — 推論のユースケース。

Inferencer Protocol と GitRepository Protocol を受け取り、
- test_path からテストデータを読み込む
- Inferencer.predict_folds() で各 fold の予測値を取得
- EnsembleStrategy で集約して最終予測を生成
- submission.csv / metainfo.yaml / README.md / .gitignore を output_dir に生成
を担う。

出力ディレクトリ構造:
  {output_dir}/
    ├── .gitignore
    ├── submission.csv
    ├── metainfo.yaml
    └── README.md

全パスは Hydra Config で管理する。
per-directory .gitignore は動的生成する（ルート .gitignore への追記禁止）。
"""

from __future__ import annotations

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

_INFERENCE_OUTPUT_GITIGNORE = """\
*
!.gitignore
!*.yaml
!*.md
!*.csv
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

    def run(self, cfg: DictConfig) -> Path:
        """推論を実行し、submission.csv を生成する。

        Args:
            cfg: Hydra DictConfig（test_path, feature_cols, models, output_dir 等を含む）

        Returns:
            submission.csv のパス
        """
        output_dir = Path(str(cfg.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        # per-directory .gitignore 生成
        gitignore_path = output_dir / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text(_INFERENCE_OUTPUT_GITIGNORE)

        # テストデータ読み込み（"latest" を含むパスを最新タイムスタンプに解決）
        test_path = resolve_latest_dir(str(cfg.test_path))
        test_df = pl.read_parquet(str(test_path))
        feature_cols: list[str] = list(cfg.feature_cols)
        passenger_id_col: str = str(cfg.get("passenger_id_col", "PassengerId"))

        # models ディレクトリのリストから予測
        # cfg.models はモデルの fold ルートディレクトリリスト（各ディレクトリに fold_N/ が含まれる）
        # ここでは単一ディレクトリを想定（複数モデルアンサンブルは models リストで指定）
        strategy = _resolve_strategy(cfg)
        predictions: list[np.ndarray] = []

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
        submission_path = output_dir / "submission.csv"
        submission.write_csv(str(submission_path))

        # metainfo.yaml に commit_hash を記録
        commit_hash = self._git_repo.get_commit_hash()
        metainfo: dict[str, Any] = {
            "job_id": str(cfg.get("job_id", "inference")),
            "commit_hash": commit_hash,
            "ensemble": str(cfg.get("ensemble", "mean")),
            "n_models": len(predictions),
            "n_test": len(test_df),
        }
        metainfo_path = output_dir / "metainfo.yaml"
        metainfo_path.write_text(yaml.dump(metainfo, allow_unicode=True, sort_keys=False))

        # README.md 生成
        self._write_readme(output_dir, metainfo)

        return submission_path

    @staticmethod
    def _write_readme(output_dir: Path, metainfo: dict[str, Any]) -> None:
        lines: list[str] = [
            f"# Inference Output — `{metainfo['job_id']}`",
            "",
            f"- commit: `{metainfo['commit_hash']}`",
            f"- ensemble: {metainfo['ensemble']}",
            f"- n_models: {metainfo['n_models']}",
            f"- n_test: {metainfo['n_test']}",
            "",
            "## Output Files",
            "",
            "```",
            output_dir.name + "/",
        ]
        lines += build_tree_lines(output_dir)
        lines.append("```")
        (output_dir / "README.md").write_text("\n".join(lines) + "\n")
