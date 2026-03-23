"""
TrainUseCase — モデル学習のユースケース。

Trainer Protocol と GitRepository Protocol を受け取り、
- preprocess_output_dir の "latest" 解決
- output_dir 生成と .gitignore 配置
- commit_hash は GitRepository 経由（subprocess 直叩き禁止）
- Trainer.fit_folds() 実行
- train_result.yaml と README.md の保存
を担う。

出力ディレクトリ構造:
  models/{competition}/{job_id}/
    ├── .gitignore
    └── {YYYYMMDDTHHMMSS}/       ← trainer が timestamp を確定、UseCase は result から使う
        ├── train_result.yaml
        ├── README.md
        ├── fold_0/
        │   ├── model.lgbm
        │   ├── oof_train.parquet
        │   ├── error_analysis.parquet
        │   └── feature_importance.parquet
        └── fold_1/ ...
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from src.domain.model.trainer import Trainer, TrainResult
from src.domain.repository.git import GitRepository
from src.usecase._utils import build_tree_lines, resolve_latest_dir

_MODELS_DIR_GITIGNORE = """\
*
!.gitignore
!*.yaml
!*.md
!*/
"""


def resolve_preprocess_dir(path_str: str) -> Path:
    """'latest' を含むパスを最新タイムスタンプディレクトリに解決する。

    NOTE: `_utils.resolve_latest_dir()` への薄いラッパー。後方互換性のために残す。

    例）
      .../titanic_preprocess/latest/train_out
      → .../titanic_preprocess/20260315T180000/train_out
    """
    return resolve_latest_dir(path_str)


class TrainUseCase:
    """Trainer と GitRepository を受け取りクロスバリデーション学習を実行する。"""

    def __init__(
        self,
        cfg: DictConfig,
        trainer: Trainer,
        git_repo: GitRepository,
    ) -> None:
        self._cfg = cfg
        self._trainer = trainer
        self._git_repo = git_repo

    def execute(self) -> TrainResult:
        cfg = self._cfg

        # preprocess_output_dir の "latest" 解決
        preprocess_dir = resolve_preprocess_dir(str(cfg.preprocess_output_dir))

        # job 出力ルート: models/{competition}/{job_id}/
        job_output_dir = Path(str(cfg.output_dir)) / str(cfg.job_id)
        job_output_dir.mkdir(parents=True, exist_ok=True)

        # .gitignore 配置（モデルバイナリ・parquet は除外、yaml/md のみ保持）
        gitignore_path = job_output_dir / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text(_MODELS_DIR_GITIGNORE)

        # timestamp と commit_hash を先に確定して trainer に渡す
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        commit_hash = self._git_repo.get_commit_hash()

        ts_dir = job_output_dir / timestamp
        ts_dir.mkdir(parents=True, exist_ok=True)

        # cfg に timestamp / commit_hash を追加して trainer へ渡す
        raw_cfg: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # ty:ignore[invalid-assignment]
        raw_cfg["_timestamp"] = timestamp
        raw_cfg["_commit_hash"] = commit_hash

        # 学習実行（fold_{N}/ は ts_dir 直下に生成される）
        result = self._trainer.fit_folds(
            preprocess_output_dir=preprocess_dir,
            output_dir=ts_dir,
            cfg=raw_cfg,
        )

        # train_result.yaml / README.md は result.timestamp ディレクトリに保存
        # (trainer が返す timestamp は cfg の _timestamp と一致する)
        result_dir = job_output_dir / result.timestamp
        result_dir.mkdir(parents=True, exist_ok=True)

        self._write_result_yaml(result_dir, result)
        self._write_readme(result_dir, result)

        return result

    @staticmethod
    def _write_result_yaml(ts_dir: Path, result: TrainResult) -> None:
        data: dict[str, Any] = {
            "job_id": result.job_id,
            "timestamp": result.timestamp,
            "commit_hash": result.commit_hash,
            "trainer_type": result.trainer_type,
            "metric": result.metric,
            "cv_mean_score": result.cv_mean_score,
            "cv_std_score": result.cv_std_score,
            "seed": result.seed,
            "folds": [
                {
                    "fold_idx": f.fold_idx,
                    "train_score": f.train_score,
                    "valid_score": f.valid_score,
                    "best_iteration": f.best_iteration,
                    "n_train": f.n_train,
                    "n_valid": f.n_valid,
                }
                for f in result.fold_results
            ],
        }
        (ts_dir / "train_result.yaml").write_text(
            yaml.dump(data, allow_unicode=True, sort_keys=False)
        )

    @staticmethod
    def _write_readme(ts_dir: Path, result: TrainResult) -> None:
        lines: list[str] = [
            f"# Train Result — `{result.job_id}`",
            "",
            f"- commit: `{result.commit_hash}`",
            f"- trainer: {result.trainer_type}",
            f"- metric: {result.metric}",
            f"- **CV score: {result.cv_mean_score:.4f} ± {result.cv_std_score:.4f}**",
            "",
            "## Fold Scores",
            "",
            f"| Fold | Train {result.metric.upper()}"
            f" | Valid {result.metric.upper()}"
            " | Best Iter | n_train | n_valid |",
            "|------|" + "----|" * 5,
        ]
        for f in result.fold_results:
            best = str(f.best_iteration) if f.best_iteration is not None else "-"
            lines.append(
                f"| {f.fold_idx} | {f.train_score:.4f} | {f.valid_score:.4f}"
                f" | {best} | {f.n_train} | {f.n_valid} |"
            )

        # ファイルツリー
        lines += [
            "",
            "## Output Files",
            "",
            "```",
            ts_dir.name + "/",
        ]
        lines += build_tree_lines(ts_dir)
        lines.append("```")

        (ts_dir / "README.md").write_text("\n".join(lines) + "\n")
