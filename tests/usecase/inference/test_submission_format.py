"""
InferenceUseCase の submission カラム名設定テスト。

なぜこのテストが必要か:
  - InferenceUseCase が target_col_name を config から受け取り、
    submission.csv のカラム名を動的に変更できることを確認する。
  - Titanic は "Survived"、Histopathologic は "label" など、
    コンペごとに異なるカラム名に対応する必要がある。

時間計算量: O(N)
空間計算量: O(N)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import polars as pl
from omegaconf import OmegaConf

from src.usecase.inference.inference import InferenceUseCase


class TestSubmissionTargetColName:
    def test_custom_target_col_name(self, tmp_path: Path) -> None:
        """target_col_name が submission.csv のカラム名に反映されること。"""
        # テストデータ
        test_path = tmp_path / "test.parquet"
        pl.DataFrame({"id": ["a", "b"], "feat": [0.5, 0.3]}).write_parquet(test_path)

        # モデルディレクトリ
        model_dir = tmp_path / "model" / "fold_0"
        model_dir.mkdir(parents=True)

        # Mock inferencer
        inferencer = MagicMock()
        inferencer.predict_folds.return_value = np.array([0.8, 0.2])

        git_repo = MagicMock()
        git_repo.get_commit_hash.return_value = "a" * 40

        cfg = OmegaConf.create(
            {
                "job_id": "test_inference",
                "test_path": str(test_path),
                "feature_cols": ["feat"],
                "passenger_id_col": "id",
                "target_col_name": "label",
                "models": [str(tmp_path / "model")],
                "ensemble": "mean",
                "output_dir": str(tmp_path / "output"),
                "submission": {"threshold": None},
            }
        )

        usecase = InferenceUseCase(inferencer=inferencer, git_repo=git_repo)
        submission_path = usecase.run(cfg)

        assert submission_path is not None
        df = pl.read_csv(str(submission_path))
        assert "label" in df.columns
        assert "Survived" not in df.columns
