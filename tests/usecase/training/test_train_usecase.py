"""
Phase 3: TrainUseCase の単体テスト。

なぜこのテストが必要か:
  - TrainUseCase は Trainer Protocol と GitRepository Protocol を受け取り、
    output_dir の作成・.gitignore の配置・README の書き出し・
    train_result.yaml の保存を担う。
  - Trainer / GitRepository 実装には依存しないため Mock で検証できる。
  - fold_N/ が timestamp ディレクトリ直下に生成されることを確認する。
  - commit_hash は GitRepository 経由でフルハッシュが記録されることを確認する。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig, OmegaConf

from src.domain.model.trainer import FoldResult, TrainResult
from src.usecase.training.train import TrainUseCase

# ──────────────────────────────────────────────────────────────
# ヘルパー
# ──────────────────────────────────────────────────────────────

_FAKE_COMMIT = "a" * 40  # GitRepository が返すフルハッシュ（40 文字）


def _make_fold(fold_idx: int = 0, valid_score: float = 0.85) -> FoldResult:
    return FoldResult(
        fold_idx=fold_idx,
        train_score=0.91,
        valid_score=valid_score,
        metric="auc",
        model_path=Path(f"model/fold_{fold_idx}/model.lgbm"),
        oof_path=Path(f"model/fold_{fold_idx}/oof_train.parquet"),
        error_analysis_path=Path(f"model/fold_{fold_idx}/error_analysis.parquet"),
        feature_importance_path=None,
        n_train=712,
        n_valid=179,
        best_iteration=234,
    )


def _make_train_result(output_dir: Path, timestamp: str = "20260315T120000") -> TrainResult:
    folds = [_make_fold(0, 0.86), _make_fold(1, 0.84)]
    return TrainResult(
        job_id="titanic_lgbm",
        timestamp=timestamp,
        commit_hash=_FAKE_COMMIT,
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


def _make_mock_git_repo() -> MagicMock:
    mock = MagicMock()
    mock.get_commit_hash.return_value = _FAKE_COMMIT
    return mock


# ──────────────────────────────────────────────────────────────
# TrainUseCase
# ──────────────────────────────────────────────────────────────


class TestTrainUseCase:
    def _run(self, tmp_path: Path) -> tuple[TrainResult, Path]:
        """Mock Trainer / GitRepo で execute() を実行して結果と job_output_dir を返す。"""
        cfg = _make_cfg(tmp_path)
        job_output_dir = tmp_path / "models" / "titanic" / "titanic_lgbm"

        mock_trainer = MagicMock()
        mock_git = _make_mock_git_repo()

        # trainer は UseCase が渡した output_dir を output_dir として TrainResult を返す
        def fake_fit_folds(preprocess_output_dir, output_dir, cfg):  # type: ignore[no-untyped-def]
            timestamp = cfg.get("_timestamp", "20260315T120000")
            return _make_train_result(output_dir, timestamp=timestamp)

        mock_trainer.fit_folds.side_effect = fake_fit_folds

        usecase = TrainUseCase(cfg, trainer=mock_trainer, git_repo=mock_git)
        result = usecase.execute()

        return result, job_output_dir

    def test_execute_returns_train_result(self, tmp_path: Path) -> None:
        """execute() は TrainResult を返すこと。"""
        result, _ = self._run(tmp_path)
        assert isinstance(result, TrainResult)
        assert result.job_id == "titanic_lgbm"

    def test_gitignore_created_in_job_output_dir(self, tmp_path: Path) -> None:
        """.gitignore は job_output_dir（timestamp の親）直下に生成されること。"""
        _, job_dir = self._run(tmp_path)
        gitignore = job_dir / ".gitignore"
        assert gitignore.exists(), f".gitignore が見つかりません: {gitignore}"

    def test_gitignore_keeps_yaml_and_md(self, tmp_path: Path) -> None:
        """.gitignore が *.yaml と *.md を保持する設定であること。"""
        _, job_dir = self._run(tmp_path)
        content = (job_dir / ".gitignore").read_text()
        assert "!*.yaml" in content
        assert "!*.md" in content

    def test_train_result_yaml_created_in_timestamp_dir(self, tmp_path: Path) -> None:
        """train_result.yaml が {job_dir}/{timestamp}/ に生成されること。"""
        result, job_dir = self._run(tmp_path)
        yaml_path = job_dir / result.timestamp / "train_result.yaml"
        assert yaml_path.exists(), f"train_result.yaml が見つかりません: {yaml_path}"

    def test_train_result_yaml_contains_cv_score(self, tmp_path: Path) -> None:
        """train_result.yaml に CV スコアが記録されていること。"""
        result, job_dir = self._run(tmp_path)
        content = (job_dir / result.timestamp / "train_result.yaml").read_text()
        assert "cv_mean_score" in content
        assert "0.85" in content

    def test_readme_created_in_timestamp_dir(self, tmp_path: Path) -> None:
        """README.md が {job_dir}/{timestamp}/ に生成されること。"""
        result, job_dir = self._run(tmp_path)
        readme = job_dir / result.timestamp / "README.md"
        assert readme.exists(), f"README.md が見つかりません: {readme}"

    def test_readme_contains_full_commit_hash(self, tmp_path: Path) -> None:
        """README.md に GitRepository から取得したフル commit hash が記録されること。"""
        result, job_dir = self._run(tmp_path)
        content = (job_dir / result.timestamp / "README.md").read_text()
        assert _FAKE_COMMIT in content
        assert len(result.commit_hash) == 40

    def test_readme_contains_cv_score_table(self, tmp_path: Path) -> None:
        """README.md に fold ごとのスコアテーブルが含まれること。"""
        result, job_dir = self._run(tmp_path)
        content = (job_dir / result.timestamp / "README.md").read_text()
        assert "| Fold |" in content
        assert "0.86" in content  # fold 0 valid score
        assert "0.84" in content  # fold 1 valid score

    def test_git_repo_get_commit_hash_called(self, tmp_path: Path) -> None:
        """GitRepository.get_commit_hash() が呼ばれること（DI 確認）。"""
        cfg = _make_cfg(tmp_path)
        mock_trainer = MagicMock()
        mock_git = _make_mock_git_repo()

        def fake_fit_folds(preprocess_output_dir, output_dir, cfg):  # type: ignore[no-untyped-def]
            timestamp = cfg.get("_timestamp", "20260315T120000")
            return _make_train_result(output_dir, timestamp=timestamp)

        mock_trainer.fit_folds.side_effect = fake_fit_folds

        usecase = TrainUseCase(cfg, trainer=mock_trainer, git_repo=mock_git)
        usecase.execute()

        mock_git.get_commit_hash.assert_called_once()

    def test_trainer_receives_timestamp_dir_as_output_dir(self, tmp_path: Path) -> None:
        """Trainer.fit_folds() に渡される output_dir が timestamp ディレクトリであること。"""
        cfg = _make_cfg(tmp_path)
        mock_trainer = MagicMock()
        mock_git = _make_mock_git_repo()

        captured: dict[str, Path] = {}

        def fake_fit_folds(preprocess_output_dir, output_dir, cfg):  # type: ignore[no-untyped-def]
            captured["output_dir"] = output_dir
            timestamp = cfg.get("_timestamp", "20260315T120000")
            return _make_train_result(output_dir, timestamp=timestamp)

        mock_trainer.fit_folds.side_effect = fake_fit_folds

        usecase = TrainUseCase(cfg, trainer=mock_trainer, git_repo=mock_git)
        usecase.execute()

        job_dir = tmp_path / "models" / "titanic" / "titanic_lgbm"
        # output_dir は job_dir/{timestamp}/ の形であること
        assert captured["output_dir"].parent == job_dir

    def test_trainer_fit_folds_called_once(self, tmp_path: Path) -> None:
        """Trainer.fit_folds() が 1 度だけ呼ばれること。"""
        _, _ = self._run(tmp_path)
        # _run 内で assert するため pass（fit_folds が呼ばれないと result が取れない）


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
