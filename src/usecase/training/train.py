"""
TrainUseCase — モデル学習のユースケース。

Trainer Protocol の実装（LightGBM / PyTorch 等）を受け取り、
- preprocess_output_dir の "latest" 解決
- output_dir 生成と .gitignore 配置
- Trainer.fit_folds() 実行
- train_result.yaml と README.md の保存
を担う。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from src.domain.model.trainer import Trainer, TrainResult

_MODELS_DIR_GITIGNORE = """\
*
!.gitignore
!*.yaml
!*.md
!*/
"""


def resolve_preprocess_dir(path_str: str) -> Path:
    """'latest' を含むパスを最新タイムスタンプディレクトリに解決する。

    例）
      .../titanic_preprocess/latest/train_out
      → .../titanic_preprocess/20260315T180000/train_out
    """
    parts = Path(path_str).parts
    latest_indices = [i for i, p in enumerate(parts) if p == "latest"]
    if not latest_indices:
        return Path(path_str)

    idx = latest_indices[0]
    parent = Path(*parts[:idx])
    suffix = Path(*parts[idx + 1 :]) if len(parts) > idx + 1 else Path()

    candidates = sorted(parent.iterdir(), key=lambda p: p.name, reverse=True)
    dirs = [c for c in candidates if c.is_dir()]
    if not dirs:
        raise ValueError(f"No processed directory found under {parent}")

    resolved = dirs[0]
    return resolved / suffix if str(suffix) != "." else resolved


class TrainUseCase:
    """Trainer を受け取りクロスバリデーション学習を実行する。"""

    def __init__(self, cfg: DictConfig, trainer: Trainer) -> None:
        self._cfg = cfg
        self._trainer = trainer

    def execute(self) -> TrainResult:
        cfg = self._cfg

        # preprocess_output_dir の "latest" 解決
        preprocess_dir = resolve_preprocess_dir(str(cfg.preprocess_output_dir))

        # output_dir = conf の output_dir / job_id
        job_output_dir = Path(str(cfg.output_dir)) / str(cfg.job_id)
        job_output_dir.mkdir(parents=True, exist_ok=True)

        # .gitignore 配置（モデルバイナリ・parquet は除外、yaml/md のみ保持）
        gitignore_path = job_output_dir / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text(_MODELS_DIR_GITIGNORE)

        # 学習実行
        result = self._trainer.fit_folds(
            preprocess_output_dir=preprocess_dir,
            output_dir=job_output_dir,
            cfg=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
        )

        # timestamp ディレクトリ作成
        ts_dir = job_output_dir / result.timestamp
        ts_dir.mkdir(parents=True, exist_ok=True)

        # train_result.yaml 保存
        self._write_result_yaml(ts_dir, result)

        # README.md 保存
        self._write_readme(ts_dir, result)

        return result

    @staticmethod
    def _get_commit_hash() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], text=True
            ).strip()
        except Exception:
            return "unknown"

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

        (ts_dir / "README.md").write_text("\n".join(lines) + "\n")
