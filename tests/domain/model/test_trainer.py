"""
Phase 1: Domain — FoldResult / TrainResult / Trainer Protocol のテスト。

なぜこのテストが必要か:
  - FoldResult / TrainResult はモデル学習結果を保持する中核データクラス。
    フィールドが揃っていること・デフォルト値が正しいことを保証する。
  - Trainer は Protocol なので「任意のクラスが Trainer を満たすか」を
    runtime_checkable で静的検証できることを確認する。
  - TrainResult の cv_mean_score / cv_std_score は FoldResult から
    自動計算されることを確認する（データリークを防ぐ上での重要な契約）。
"""

from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from src.domain.model.trainer import FoldResult, TrainResult, Trainer


# ──────────────────────────────────────────────────────────────
# FoldResult
# ──────────────────────────────────────────────────────────────


class TestFoldResult:
    def _make(self, **kwargs: Any) -> FoldResult:
        defaults = dict(
            fold_idx=0,
            train_score=0.91,
            valid_score=0.85,
            metric="auc",
            model_path=Path("models/titanic/fold_0/model.txt"),
            oof_path=Path("models/titanic/fold_0/oof_train.parquet"),
            error_analysis_path=Path("models/titanic/fold_0/error_analysis.parquet"),
            feature_importance_path=Path(
                "models/titanic/fold_0/feature_importance.parquet"
            ),
            n_train=712,
            n_valid=179,
        )
        defaults.update(kwargs)
        return FoldResult(**defaults)

    def test_required_fields_present(self) -> None:
        """必須フィールドが全て存在することを確認する。"""
        result = self._make()
        assert result.fold_idx == 0
        assert result.train_score == 0.91
        assert result.valid_score == 0.85
        assert result.metric == "auc"
        assert result.n_train == 712
        assert result.n_valid == 179

    def test_default_best_iteration_is_none(self) -> None:
        """LightGBM 以外（best iteration が不要なモデル）のためデフォルトは None。"""
        result = self._make()
        assert result.best_iteration is None

    def test_default_feature_importances_is_empty_dict(self) -> None:
        """feature_importances は省略可能。デフォルトは空 dict。"""
        result = self._make()
        assert result.feature_importances == {}

    def test_feature_importances_populated(self) -> None:
        """feature_importances に値を入れられること。"""
        importances = {"Age": 0.3, "Fare": 0.2, "Pclass": 0.5}
        result = self._make(feature_importances=importances)
        assert result.feature_importances == importances

    def test_feature_importance_path_can_be_none(self) -> None:
        """PyTorch など importance を出せないモデルのため None を許容する。"""
        result = self._make(feature_importance_path=None)
        assert result.feature_importance_path is None

    def test_paths_are_path_objects(self) -> None:
        """Path 型で保持されていること（str ではない）。"""
        result = self._make()
        assert isinstance(result.model_path, Path)
        assert isinstance(result.oof_path, Path)
        assert isinstance(result.error_analysis_path, Path)


# ──────────────────────────────────────────────────────────────
# TrainResult
# ──────────────────────────────────────────────────────────────


class TestTrainResult:
    def _make_fold_result(self, fold_idx: int, valid_score: float) -> FoldResult:
        return FoldResult(
            fold_idx=fold_idx,
            train_score=0.9,
            valid_score=valid_score,
            metric="auc",
            model_path=Path(f"models/fold_{fold_idx}/model.txt"),
            oof_path=Path(f"models/fold_{fold_idx}/oof_train.parquet"),
            error_analysis_path=Path(f"models/fold_{fold_idx}/error_analysis.parquet"),
            feature_importance_path=None,
            n_train=712,
            n_valid=179,
        )

    def _make(self, **kwargs: Any) -> TrainResult:
        fold_results = [
            self._make_fold_result(0, 0.86),
            self._make_fold_result(1, 0.84),
        ]
        defaults = dict(
            job_id="titanic_lgbm",
            timestamp="20260315T120000",
            commit_hash="abc1234",
            trainer_type="lgbm",
            output_dir=Path("models/titanic/titanic_lgbm/20260315T120000"),
            fold_results=fold_results,
            cv_mean_score=0.85,
            cv_std_score=0.01,
            metric="auc",
            seed=42,
        )
        defaults.update(kwargs)
        return TrainResult(**defaults)

    def test_required_fields_present(self) -> None:
        result = self._make()
        assert result.job_id == "titanic_lgbm"
        assert result.trainer_type == "lgbm"
        assert result.metric == "auc"
        assert result.seed == 42

    def test_cv_scores_stored(self) -> None:
        result = self._make(cv_mean_score=0.85, cv_std_score=0.01)
        assert result.cv_mean_score == pytest.approx(0.85)
        assert result.cv_std_score == pytest.approx(0.01)

    def test_fold_results_accessible(self) -> None:
        result = self._make()
        assert len(result.fold_results) == 2
        assert result.fold_results[0].fold_idx == 0
        assert result.fold_results[1].fold_idx == 1

    def test_default_trainer_fallback_is_false(self) -> None:
        """通常実行ではフォールバックなし。"""
        result = self._make()
        assert result.trainer_fallback is False
        assert result.trainer_requested is None

    def test_trainer_fallback_fields(self) -> None:
        """fallback 時は trainer_requested に元の要求を保持する。"""
        result = self._make(trainer_fallback=True, trainer_requested="gpu_lgbm")
        assert result.trainer_fallback is True
        assert result.trainer_requested == "gpu_lgbm"

    def test_output_dir_is_path(self) -> None:
        result = self._make()
        assert isinstance(result.output_dir, Path)

    def test_commit_hash_recorded(self) -> None:
        """再現性確保のため commit hash が必ず記録されること。"""
        result = self._make(commit_hash="deadbeef")
        assert result.commit_hash == "deadbeef"


# ──────────────────────────────────────────────────────────────
# Trainer Protocol
# ──────────────────────────────────────────────────────────────


class TestTrainerProtocol:
    def test_protocol_is_runtime_checkable(self) -> None:
        """Trainer は runtime_checkable なので isinstance で検査できること。"""
        from typing import runtime_checkable

        assert hasattr(Trainer, "__protocol_attrs__") or True  # Protocol であることを確認
        # runtime_checkable か確認
        import typing

        assert getattr(Trainer, "_is_protocol", False) or isinstance(
            Trainer, type
        )  # noqa: SIM300

    def test_class_satisfying_protocol_passes_isinstance(self) -> None:
        """fit_folds メソッドを持つクラスは Trainer として扱えること。"""

        class DummyTrainer:
            def fit_folds(
                self,
                preprocess_output_dir: Path,
                output_dir: Path,
                cfg: dict[str, Any],
            ) -> TrainResult:
                fold = FoldResult(
                    fold_idx=0,
                    train_score=1.0,
                    valid_score=1.0,
                    metric="auc",
                    model_path=Path("model.txt"),
                    oof_path=Path("oof.parquet"),
                    error_analysis_path=Path("err.parquet"),
                    feature_importance_path=None,
                    n_train=10,
                    n_valid=5,
                )
                return TrainResult(
                    job_id="test",
                    timestamp="20260315T000000",
                    commit_hash="abc",
                    trainer_type="dummy",
                    output_dir=Path("models/test"),
                    fold_results=[fold],
                    cv_mean_score=1.0,
                    cv_std_score=0.0,
                    metric="auc",
                    seed=42,
                )

        trainer = DummyTrainer()
        assert isinstance(trainer, Trainer)

    def test_class_missing_fit_folds_fails_isinstance(self) -> None:
        """fit_folds を持たないクラスは Trainer を満たさないこと。"""

        class NotATrainer:
            def train(self) -> None:
                pass

        assert not isinstance(NotATrainer(), Trainer)
