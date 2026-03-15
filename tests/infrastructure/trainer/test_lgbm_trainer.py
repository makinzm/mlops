"""
Phase 4: LightGBMTrainer の統合テスト。

なぜこのテストが必要か:
  - LightGBMTrainer は fold ごとに実際に lgb.train() を呼ぶため、
    fold 構造の parquet を読み込んで学習・OOF 予測・ファイル保存が
    正しく動くことを結合テストで確認する。
  - error_analysis.parquet の 4 分類サンプリングが正しく動くことを確認する。
  - feature_importance.parquet が保存されることを確認する。
  - サンプル重み（sample_weight_col / class_weight / is_unbalance）の
    優先順位が正しいことを確認する。
  - output は input と異なる Dir であることを確認する（再現性）。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from omegaconf import OmegaConf

from src.domain.model.trainer import TrainResult
from src.infrastructure.trainer.lgbm_trainer import LightGBMTrainer


# ──────────────────────────────────────────────────────────────
# フィクスチャ
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def fold_dir(tmp_path: Path) -> Path:
    """fold_0/ 配下に train.parquet / test.parquet を持つ前処理出力ディレクトリ。"""
    fold0 = tmp_path / "preprocess_out" / "fold_0"
    fold0.mkdir(parents=True)

    # 二値分類タスク: Pclass, Age, Fare → Survived
    import numpy as np

    rng = np.random.default_rng(42)
    n_train, n_valid = 100, 30

    train_df = pl.DataFrame(
        {
            "PassengerId": list(range(1, n_train + 1)),
            "Survived": rng.integers(0, 2, n_train).tolist(),
            "Pclass": rng.integers(1, 4, n_train).tolist(),
            "Age": rng.uniform(1, 80, n_train).tolist(),
            "Fare": rng.uniform(5, 500, n_train).tolist(),
            "Weight": rng.uniform(0.5, 2.0, n_train).tolist(),
        }
    )
    valid_df = pl.DataFrame(
        {
            "PassengerId": list(range(n_train + 1, n_train + n_valid + 1)),
            "Survived": rng.integers(0, 2, n_valid).tolist(),
            "Pclass": rng.integers(1, 4, n_valid).tolist(),
            "Age": rng.uniform(1, 80, n_valid).tolist(),
            "Fare": rng.uniform(5, 500, n_valid).tolist(),
            "Weight": rng.uniform(0.5, 2.0, n_valid).tolist(),
        }
    )

    train_df.write_parquet(fold0 / "train.parquet")
    valid_df.write_parquet(fold0 / "test.parquet")
    return tmp_path / "preprocess_out"


@pytest.fixture
def base_cfg(tmp_path: Path) -> dict:
    return {
        "job_id": "test_lgbm",
        "target_col": "Survived",
        "feature_cols": ["Pclass", "Age", "Fare"],
        "categorical_feature": [],
        "sample_weight_col": None,
        "loss": {
            "objective": "binary",
            "metric": "auc",
            "is_unbalance": False,
            "class_weight": None,
        },
        "lgbm": {
            "num_leaves": 8,
            "learning_rate": 0.1,
            "n_estimators": 20,
            "verbose": -1,
            "early_stopping_rounds": 5,
        },
        "report": {"n_error_samples": 5},
        "environment": {"device": "cpu", "n_jobs": 1},
        "logging": {"eval_freq": 10, "save_importance": True},
        "seed": 42,
    }


# ──────────────────────────────────────────────────────────────
# fit_folds
# ──────────────────────────────────────────────────────────────


class TestLightGBMTrainerFitFolds:
    def _run(self, fold_dir: Path, tmp_path: Path, cfg_dict: dict) -> TrainResult:
        output_dir = tmp_path / "models" / "test_lgbm"
        cfg = OmegaConf.create(cfg_dict)
        trainer = LightGBMTrainer(cfg)
        return trainer.fit_folds(
            preprocess_output_dir=fold_dir,
            output_dir=output_dir,
            cfg=cfg_dict,
        )

    def test_returns_train_result(
        self, fold_dir: Path, tmp_path: Path, base_cfg: dict
    ) -> None:
        """fit_folds は TrainResult を返すこと。"""
        result = self._run(fold_dir, tmp_path, base_cfg)
        assert isinstance(result, TrainResult)
        assert result.trainer_type == "lgbm"
        assert result.metric == "auc"

    def test_fold_results_count_matches_folds(
        self, fold_dir: Path, tmp_path: Path, base_cfg: dict
    ) -> None:
        """fold_results の数が実際の fold 数と一致すること。"""
        result = self._run(fold_dir, tmp_path, base_cfg)
        # fold_0/ のみ作成したので 1 fold
        assert len(result.fold_results) == 1
        assert result.fold_results[0].fold_idx == 0

    def test_cv_scores_are_computed(
        self, fold_dir: Path, tmp_path: Path, base_cfg: dict
    ) -> None:
        """cv_mean_score / cv_std_score が計算されていること。"""
        result = self._run(fold_dir, tmp_path, base_cfg)
        assert 0.0 < result.cv_mean_score <= 1.0
        assert result.cv_std_score >= 0.0

    def test_model_file_saved(
        self, fold_dir: Path, tmp_path: Path, base_cfg: dict
    ) -> None:
        """model.txt が fold ディレクトリに保存されること。"""
        result = self._run(fold_dir, tmp_path, base_cfg)
        assert result.fold_results[0].model_path.exists()

    def test_oof_parquet_saved(
        self, fold_dir: Path, tmp_path: Path, base_cfg: dict
    ) -> None:
        """oof_train.parquet が保存されること。"""
        result = self._run(fold_dir, tmp_path, base_cfg)
        assert result.fold_results[0].oof_path.exists()

    def test_error_analysis_parquet_saved(
        self, fold_dir: Path, tmp_path: Path, base_cfg: dict
    ) -> None:
        """error_analysis.parquet が保存されること。"""
        result = self._run(fold_dir, tmp_path, base_cfg)
        assert result.fold_results[0].error_analysis_path.exists()

    def test_error_analysis_has_required_columns(
        self, fold_dir: Path, tmp_path: Path, base_cfg: dict
    ) -> None:
        """error_analysis.parquet に必要なカラムが含まれること。"""
        result = self._run(fold_dir, tmp_path, base_cfg)
        df = pd.read_parquet(result.fold_results[0].error_analysis_path)
        for col in ["target", "predicted_proba", "predicted_label", "is_correct",
                    "error_magnitude", "sample_type"]:
            assert col in df.columns, f"カラム '{col}' が error_analysis に存在しません"

    def test_error_analysis_sample_types(
        self, fold_dir: Path, tmp_path: Path, base_cfg: dict
    ) -> None:
        """sample_type は TP/TN/FP/FN の 4 種のいずれかであること。"""
        result = self._run(fold_dir, tmp_path, base_cfg)
        df = pd.read_parquet(result.fold_results[0].error_analysis_path)
        valid_types = {"TP", "TN", "FP", "FN"}
        assert set(df["sample_type"].unique()).issubset(valid_types)

    def test_feature_importance_saved_when_enabled(
        self, fold_dir: Path, tmp_path: Path, base_cfg: dict
    ) -> None:
        """save_importance=True のとき feature_importance.parquet が保存されること。"""
        result = self._run(fold_dir, tmp_path, base_cfg)
        fi_path = result.fold_results[0].feature_importance_path
        assert fi_path is not None
        assert fi_path.exists()

    def test_output_dir_differs_from_input_dir(
        self, fold_dir: Path, tmp_path: Path, base_cfg: dict
    ) -> None:
        """output_dir と preprocess_output_dir が異なること（再現性保証）。"""
        output_dir = tmp_path / "models" / "test_lgbm"
        cfg = OmegaConf.create(base_cfg)
        trainer = LightGBMTrainer(cfg)
        trainer.fit_folds(
            preprocess_output_dir=fold_dir,
            output_dir=output_dir,
            cfg=base_cfg,
        )
        assert fold_dir.resolve() != output_dir.resolve()

    def test_seed_reproducibility(
        self, fold_dir: Path, tmp_path: Path, base_cfg: dict
    ) -> None:
        """同じシードで2回実行したとき CV スコアが一致すること。"""
        result1 = self._run(fold_dir, tmp_path / "run1", base_cfg)
        result2 = self._run(fold_dir, tmp_path / "run2", base_cfg)
        assert result1.cv_mean_score == pytest.approx(result2.cv_mean_score, abs=1e-6)


class TestSampleWeights:
    def test_sample_weight_col_used_when_specified(
        self, fold_dir: Path, tmp_path: Path, base_cfg: dict
    ) -> None:
        """sample_weight_col を指定したとき学習が正常に完了すること。"""
        cfg_dict = dict(base_cfg)
        cfg_dict["sample_weight_col"] = "Weight"
        cfg = OmegaConf.create(cfg_dict)
        trainer = LightGBMTrainer(cfg)
        result = trainer.fit_folds(
            preprocess_output_dir=fold_dir,
            output_dir=tmp_path / "models",
            cfg=cfg_dict,
        )
        assert isinstance(result, TrainResult)
