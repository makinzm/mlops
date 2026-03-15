"""
InferenceUseCase の単体テスト。

なぜこのテストが必要か:
  - InferenceUseCase は Inferencer Protocol と EnsembleStrategy を受け取り、
    submission.csv / metainfo.yaml / README.md / .gitignore を生成する責務を持つ。
  - 実際の LightGBM モデルに依存せず Mock で検証することで、
    UseCase の責務（ファイル生成・設定読み込み・commit_hash 記録）を
    インフラから独立してテストできる。
  - submission.csv の PassengerId 列と予測値列の存在を確認することで、
    Kaggle 提出フォーマットへの準拠を保証できる。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest
from omegaconf import DictConfig, OmegaConf

from src.usecase.inference.inference import InferenceUseCase

# ──────────────────────────────────────────────────────────────
# ヘルパー
# ──────────────────────────────────────────────────────────────

_FAKE_COMMIT = "b" * 40  # GitRepository が返すフルハッシュ


def _make_test_parquet(path: Path, n: int = 5) -> pl.DataFrame:
    """テスト用 test.parquet を生成して path に保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "PassengerId": list(range(892, 892 + n)),
            "Pclass": [1, 2, 3, 1, 2],
            "Age": [30.0, 25.0, 40.0, 35.0, 28.0],
        }
    )
    df.write_parquet(path)
    return df


def _make_cfg(
    tmp_path: Path,
    *,
    ensemble: str = "mean",
    weights: list[float] | None = None,
    n_models: int = 2,
) -> DictConfig:
    """InferenceUseCase 用の最小 DictConfig を生成する。"""
    test_parquet = str(tmp_path / "processed" / "test.parquet")
    models = [str(tmp_path / f"models/fold_{i}/model.lgbm") for i in range(n_models)]

    raw: dict[str, object] = {
        "job_id": "titanic_inference",
        "output_dir": str(tmp_path / "inference_out"),
        "test_path": test_parquet,
        "feature_cols": ["Pclass", "Age"],
        "passenger_id_col": "PassengerId",
        "models": models,
        "ensemble": ensemble,
        "seed": 42,
    }
    if weights is not None:
        raw["weights"] = weights
    return OmegaConf.create(raw)


@pytest.fixture
def mock_inferencer() -> MagicMock:
    """Inferencer Protocol のモック。predict_folds は固定の ndarray を返す。"""
    mock = MagicMock()
    # 5 サンプルの予測値
    mock.predict_folds.return_value = np.array([0.1, 0.8, 0.3, 0.9, 0.5])
    return mock


@pytest.fixture
def mock_git_repo() -> MagicMock:
    """GitRepository Protocol のモック。"""
    mock = MagicMock()
    mock.get_commit_hash.return_value = _FAKE_COMMIT
    return mock


class TestInferenceUseCaseRun:
    def test_run_creates_submission_csv(
        self, tmp_path: Path, mock_inferencer: MagicMock, mock_git_repo: MagicMock
    ) -> None:
        """InferenceUseCase.run() が submission.csv を生成すること。"""
        cfg = _make_cfg(tmp_path)
        _make_test_parquet(Path(cfg.test_path))

        usecase = InferenceUseCase(inferencer=mock_inferencer, git_repo=mock_git_repo)
        usecase.run(cfg)

        submission_path = Path(cfg.output_dir) / "submission.csv"
        assert submission_path.exists(), f"submission.csv が生成されていない: {submission_path}"

    def test_run_submission_has_correct_columns(
        self, tmp_path: Path, mock_inferencer: MagicMock, mock_git_repo: MagicMock
    ) -> None:
        """submission.csv が PassengerId と Survived 列を持つこと（Kaggle フォーマット）。"""
        cfg = _make_cfg(tmp_path)
        _make_test_parquet(Path(cfg.test_path))

        usecase = InferenceUseCase(inferencer=mock_inferencer, git_repo=mock_git_repo)
        usecase.run(cfg)

        submission = pl.read_csv(Path(cfg.output_dir) / "submission.csv")
        assert "PassengerId" in submission.columns
        assert "Survived" in submission.columns
        assert len(submission) == 5  # _make_test_parquet の n=5

    def test_run_uses_ensemble_strategy(
        self, tmp_path: Path, mock_inferencer: MagicMock, mock_git_repo: MagicMock
    ) -> None:
        """MockInferencer の予測値が submission に反映されること（strategy が適用されること）。"""
        cfg = _make_cfg(tmp_path, ensemble="mean")
        _make_test_parquet(Path(cfg.test_path))

        mock_inferencer.predict_folds.return_value = np.array([0.1, 0.8, 0.3, 0.9, 0.5])

        usecase = InferenceUseCase(inferencer=mock_inferencer, git_repo=mock_git_repo)
        usecase.run(cfg)

        # predict_folds が少なくとも 1 回呼ばれたことを確認（models リストの件数分呼ばれる）
        mock_inferencer.predict_folds.assert_called()

    def test_run_records_commit_hash(
        self, tmp_path: Path, mock_inferencer: MagicMock, mock_git_repo: MagicMock
    ) -> None:
        """output の metainfo.yaml に commit_hash が記録されること。"""
        import yaml

        cfg = _make_cfg(tmp_path)
        _make_test_parquet(Path(cfg.test_path))

        usecase = InferenceUseCase(inferencer=mock_inferencer, git_repo=mock_git_repo)
        usecase.run(cfg)

        metainfo_path = Path(cfg.output_dir) / "metainfo.yaml"
        assert metainfo_path.exists(), f"metainfo.yaml が生成されていない: {metainfo_path}"

        with open(metainfo_path) as f:
            meta = yaml.safe_load(f)
        assert "commit_hash" in meta
        assert meta["commit_hash"] == _FAKE_COMMIT

    def test_run_generates_gitignore(
        self, tmp_path: Path, mock_inferencer: MagicMock, mock_git_repo: MagicMock
    ) -> None:
        """per-directory .gitignore が output_dir に生成されること。"""
        cfg = _make_cfg(tmp_path)
        _make_test_parquet(Path(cfg.test_path))

        usecase = InferenceUseCase(inferencer=mock_inferencer, git_repo=mock_git_repo)
        usecase.run(cfg)

        gitignore_path = Path(cfg.output_dir) / ".gitignore"
        assert gitignore_path.exists(), f".gitignore が生成されていない: {gitignore_path}"

    def test_run_skips_submission_when_no_test_path(
        self, tmp_path: Path, mock_inferencer: MagicMock, mock_git_repo: MagicMock
    ) -> None:
        """test_path=None のとき submission.csv をスキップし metainfo.yaml / README.md は生成する。

        なぜこのテストが必要か:
          - test データなしコンペ（推論先が存在しない場合）でも usecase が crash せずに
            metainfo.yaml / README.md だけ出力できることを保証する。
          - Kaggle の一部コンペは test.csv を提供しないため、null チェックは必須。
        """
        cfg = _make_cfg(tmp_path)
        # test_path を None に上書き
        raw = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(raw, dict)
        raw["test_path"] = None
        cfg = OmegaConf.create(raw)

        usecase = InferenceUseCase(inferencer=mock_inferencer, git_repo=mock_git_repo)
        usecase.run(cfg)

        submission_path = Path(cfg.output_dir) / "submission.csv"
        assert not submission_path.exists(), (
            "test_path=None でも submission.csv が生成されてしまった"
        )

        metainfo_path = Path(cfg.output_dir) / "metainfo.yaml"
        assert metainfo_path.exists(), "metainfo.yaml が生成されていない"

        readme_path = Path(cfg.output_dir) / "README.md"
        assert readme_path.exists(), "README.md が生成されていない"

    def test_run_skips_submission_when_test_file_missing(
        self, tmp_path: Path, mock_inferencer: MagicMock, mock_git_repo: MagicMock
    ) -> None:
        """test_path が指定されていてもファイル不在の場合は submission.csv をスキップすること。

        なぜこのテストが必要か:
          - パス設定ミスや前処理が未実行の場合に usecase が crash すると
            pipeline 全体が止まってしまう。
          - ファイル不在を graceful に扱い、metainfo.yaml / README.md だけ出力することで
            後から原因を特定しやすくなる。
        """
        cfg = _make_cfg(tmp_path)
        # test_path は設定されているが、ファイルを作成しない

        usecase = InferenceUseCase(inferencer=mock_inferencer, git_repo=mock_git_repo)
        usecase.run(cfg)

        submission_path = Path(cfg.output_dir) / "submission.csv"
        assert not submission_path.exists(), (
            "存在しない test_path でも submission.csv が生成されてしまった"
        )

        metainfo_path = Path(cfg.output_dir) / "metainfo.yaml"
        assert metainfo_path.exists(), "metainfo.yaml が生成されていない"

        readme_path = Path(cfg.output_dir) / "README.md"
        assert readme_path.exists(), "README.md が生成されていない"


class TestInferenceUseCaseTimestamp:
    """タイムスタンプ付き出力ディレクトリ構造のテスト。

    なぜこのテストが必要か:
      - train/preprocess と同様に {output_dir}/{job_id}/{YYYYMMDDTHHMMSS}/ 形式で
        出力ディレクトリを作成することで、実行ごとの結果が上書きされず
        過去の推論結果を追跡できることを保証する。
      - latest シンボリックリンク解決や pipeline usecase での連携にも必要。
    """

    def test_run_output_dir_has_timestamp_subdir(
        self, tmp_path: Path, mock_inferencer: MagicMock, mock_git_repo: MagicMock
    ) -> None:
        """output_dir/{job_id}/{timestamp}/ サブディレクトリが生成されること。

        現行実装は {output_dir}/ 直下にファイルを生成するため、このテストは FAIL する。
        """
        cfg = _make_cfg(tmp_path)
        _make_test_parquet(Path(cfg.test_path))

        usecase = InferenceUseCase(inferencer=mock_inferencer, git_repo=mock_git_repo)
        usecase.run(cfg)

        job_dir = Path(cfg.output_dir) / str(cfg.job_id)
        assert job_dir.exists(), f"job_id ディレクトリが生成されていない: {job_dir}"

        # job_dir 直下に YYYYMMDDTHHMMSS 形式のサブディレクトリが存在すること
        ts_dirs = [d for d in job_dir.iterdir() if d.is_dir() and len(d.name) == 15]
        assert len(ts_dirs) >= 1, (
            f"{job_dir} 直下にタイムスタンプサブディレクトリが生成されていない"
        )

    def test_run_returns_submission_path_in_timestamp_dir(
        self, tmp_path: Path, mock_inferencer: MagicMock, mock_git_repo: MagicMock
    ) -> None:
        """run() の戻り値が {output_dir}/{job_id}/{timestamp}/submission.csv であること。"""
        cfg = _make_cfg(tmp_path)
        _make_test_parquet(Path(cfg.test_path))

        usecase = InferenceUseCase(inferencer=mock_inferencer, git_repo=mock_git_repo)
        result = usecase.run(cfg)

        assert result is not None, "submission.csv のパスが None"
        assert result.exists(), f"submission.csv が存在しない: {result}"

        # パスが {output_dir}/{job_id}/{timestamp}/submission.csv であること
        assert result.name == "submission.csv"
        # timestamp ディレクトリは result.parent.name、job_id は result.parent.parent.name
        ts_name = result.parent.name
        assert len(ts_name) == 15, (
            f"timestamp ディレクトリ名が YYYYMMDDTHHMMSS 形式でない: {ts_name}"
        )
        assert result.parent.parent.name == str(cfg.job_id), (
            f"job_id ディレクトリ名が一致しない: {result.parent.parent.name}"
        )

    def test_run_metainfo_records_timestamp(
        self, tmp_path: Path, mock_inferencer: MagicMock, mock_git_repo: MagicMock
    ) -> None:
        """metainfo.yaml に timestamp が記録されること。"""
        import yaml

        cfg = _make_cfg(tmp_path)
        _make_test_parquet(Path(cfg.test_path))

        usecase = InferenceUseCase(inferencer=mock_inferencer, git_repo=mock_git_repo)
        result = usecase.run(cfg)

        assert result is not None
        metainfo_path = result.parent / "metainfo.yaml"
        assert metainfo_path.exists(), f"metainfo.yaml が生成されていない: {metainfo_path}"

        with open(metainfo_path) as f:
            meta = yaml.safe_load(f)
        assert "timestamp" in meta, "metainfo.yaml に timestamp が記録されていない"
        assert len(meta["timestamp"]) == 15, (
            f"timestamp が YYYYMMDDTHHMMSS 形式でない: {meta['timestamp']}"
        )
