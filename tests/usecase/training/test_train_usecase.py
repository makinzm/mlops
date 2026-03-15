"""
Phase 3: TrainUseCase の単体テスト。

なぜこのテストが必要か:
  - TrainUseCase は Trainer Protocol を受け取り、
    output_dir の作成・.gitignore の配置・README の書き出し・
    train_result.yaml の保存を担う。
  - Trainer 実装（LightGBM / PyTorch）には依存しないため Mock で検証できる。
  - ファイル生成・パス解決を個別にテストすることで、
    後から Trainer 実装を差し替えても動作することを保証する。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig, OmegaConf

from src.domain.model.trainer import FoldResult, TrainResult
from src.usecase.training.train import TrainUseCase

# ──────────────────────────────────────────────────────────────
# ヘルパー
# ──────────────────────────────────────────────────────────────


def _make_fold(fold_idx: int = 0, valid_score: float = 0.85) -> FoldResult:
    return FoldResult(
        fold_idx=fold_idx,
        train_score=0.91,
        valid_score=valid_score,
        metric="auc",
        model_path=Path(f"model/fold_{fold_idx}/model.txt"),
        oof_path=Path(f"model/fold_{fold_idx}/oof_train.parquet"),
        error_analysis_path=Path(f"model/fold_{fold_idx}/error_analysis.parquet"),
        feature_importance_path=None,
        n_train=712,
        n_valid=179,
        best_iteration=234,
    )


def _make_train_result(output_dir: Path) -> TrainResult:
    folds = [_make_fold(0, 0.86), _make_fold(1, 0.84)]
    return TrainResult(
        job_id="titanic_lgbm",
        timestamp="20260315T120000",
        commit_hash="abc1234",
        trainer_type="lgbm",
        output_dir=output_dir,
        fold_results=folds,
        cv_mean_score=0.85,
        cv_std_score=0.01,
        metric="auc",
        seed=42,
    )


def _make_cfg(tmp_path: Path) -> DictConfig:
    return OmegaConf.create(
        {
            "job_id": "titanic_lgbm",
            "competition": {"name": "titanic"},
            "preprocess_output_dir": str(tmp_path / "processed"),
            "target_col": "Survived",
            "feature_cols": ["Pclass", "Age", "Fare"],
            "trainer": {"type": "lgbm"},
            "output_dir": str(tmp_path / "models" / "titanic"),
            "seed": 42,
        }
    )


# ──────────────────────────────────────────────────────────────
# TrainUseCase
# ──────────────────────────────────────────────────────────────


class TestTrainUseCase:
    def _run(self, tmp_path: Path) -> tuple[TrainResult, Path]:
        """TrainUseCase.execute() を Mock Trainer で実行して結果と output_dir を返す。"""
        cfg = _make_cfg(tmp_path)

        expected_output_dir = tmp_path / "models" / "titanic" / "titanic_lgbm"
        train_result = _make_train_result(expected_output_dir)

        mock_trainer = MagicMock()
        mock_trainer.fit_folds.return_value = train_result

        with patch(
            "src.usecase.training.train.TrainUseCase._get_commit_hash",
            return_value="abc1234",
        ):
            usecase = TrainUseCase(cfg, trainer=mock_trainer)
            result = usecase.execute()

        return result, expected_output_dir

    def test_execute_returns_train_result(self, tmp_path: Path) -> None:
        """execute() は TrainResult を返すこと。"""
        result, _ = self._run(tmp_path)
        assert isinstance(result, TrainResult)
        assert result.job_id == "titanic_lgbm"

    def test_gitignore_created_in_output_dir(self, tmp_path: Path) -> None:
        """output_dir 直下に .gitignore が生成されること。"""
        _, out_dir = self._run(tmp_path)
        gitignore = out_dir / ".gitignore"
        assert gitignore.exists(), f".gitignore が見つかりません: {gitignore}"

    def test_gitignore_keeps_yaml_and_md(self, tmp_path: Path) -> None:
        """.gitignore が *.yaml と *.md を保持する設定であること。"""
        _, out_dir = self._run(tmp_path)
        content = (out_dir / ".gitignore").read_text()
        assert "!*.yaml" in content
        assert "!*.md" in content

    def test_train_result_yaml_created(self, tmp_path: Path) -> None:
        """train_result.yaml が output_dir 直下に生成されること。"""
        result, out_dir = self._run(tmp_path)
        yaml_path = out_dir / result.timestamp / "train_result.yaml"
        assert yaml_path.exists(), f"train_result.yaml が見つかりません: {yaml_path}"

    def test_train_result_yaml_contains_cv_score(self, tmp_path: Path) -> None:
        """train_result.yaml に CV スコアが記録されていること。"""
        result, out_dir = self._run(tmp_path)
        content = (out_dir / result.timestamp / "train_result.yaml").read_text()
        assert "cv_mean_score" in content
        assert "0.85" in content

    def test_readme_created_in_timestamp_dir(self, tmp_path: Path) -> None:
        """README.md が output_dir/{timestamp}/ に生成されること。"""
        result, out_dir = self._run(tmp_path)
        readme = out_dir / result.timestamp / "README.md"
        assert readme.exists(), f"README.md が見つかりません: {readme}"

    def test_readme_contains_commit_hash(self, tmp_path: Path) -> None:
        """README.md に commit hash が記録されていること（再現性保証）。"""
        result, out_dir = self._run(tmp_path)
        content = (out_dir / result.timestamp / "README.md").read_text()
        assert result.commit_hash in content

    def test_readme_contains_cv_score_table(self, tmp_path: Path) -> None:
        """README.md に fold ごとのスコアテーブルが含まれること。"""
        result, out_dir = self._run(tmp_path)
        content = (out_dir / result.timestamp / "README.md").read_text()
        assert "| Fold |" in content
        assert "0.86" in content  # fold 0 valid score
        assert "0.84" in content  # fold 1 valid score

    def test_trainer_fit_folds_called_once(self, tmp_path: Path) -> None:
        """Trainer.fit_folds() が 1 度だけ呼ばれること。"""
        cfg = _make_cfg(tmp_path)
        expected_output_dir = tmp_path / "models" / "titanic" / "titanic_lgbm"
        train_result = _make_train_result(expected_output_dir)

        mock_trainer = MagicMock()
        mock_trainer.fit_folds.return_value = train_result

        with patch(
            "src.usecase.training.train.TrainUseCase._get_commit_hash",
            return_value="abc1234",
        ):
            usecase = TrainUseCase(cfg, trainer=mock_trainer)
            usecase.execute()

        mock_trainer.fit_folds.assert_called_once()


# ──────────────────────────────────────────────────────────────
# preprocess_output_dir の "latest" 解決
# ──────────────────────────────────────────────────────────────


class TestResolveLatestDir:
    def test_latest_resolves_to_newest_timestamp(self, tmp_path: Path) -> None:
        """'latest' を含むパスが最新タイムスタンプに解決されること。"""
        from src.usecase.training.train import resolve_preprocess_dir

        base = tmp_path / "processed" / "titanic_preprocess"
        (base / "20260314T120000").mkdir(parents=True)
        (base / "20260315T180000").mkdir(parents=True)
        (base / "20260313T000000").mkdir(parents=True)

        resolved = resolve_preprocess_dir(str(base / "latest"))
        assert resolved == base / "20260315T180000"

    def test_latest_with_suffix_resolves_correctly(self, tmp_path: Path) -> None:
        """'latest/train_out' のようなサフィックスも正しく解決されること。"""
        from src.usecase.training.train import resolve_preprocess_dir

        base = tmp_path / "processed" / "titanic_preprocess"
        train_out = base / "20260315T180000" / "train_out"
        train_out.mkdir(parents=True)

        resolved = resolve_preprocess_dir(str(base / "latest" / "train_out"))
        assert resolved == train_out

    def test_no_latest_returns_path_as_is(self, tmp_path: Path) -> None:
        """'latest' を含まないパスはそのまま返すこと。"""
        from src.usecase.training.train import resolve_preprocess_dir

        path = tmp_path / "processed" / "20260315T180000" / "train_out"
        resolved = resolve_preprocess_dir(str(path))
        assert resolved == path

    def test_raises_when_no_timestamp_dirs(self, tmp_path: Path) -> None:
        """latest 配下にディレクトリがない場合は ValueError。"""
        from src.usecase.training.train import resolve_preprocess_dir

        base = tmp_path / "processed"
        base.mkdir(parents=True)

        with pytest.raises(ValueError, match="No processed directory"):
            resolve_preprocess_dir(str(base / "latest"))
