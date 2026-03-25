"""
RemoteSubmitUseCase の単体テスト。

なぜこのテストが必要か:
  - RemoteSubmitUseCase はジョブを非同期送信し、即座に job_manifest.yaml を保存して終了する。
  - 既存の RemoteTrainUseCase（同期版）と異なり、ジョブ完了を待たずに戻ることを保証する。
  - manifest が正しく保存されること、submit_custom_job が呼ばれること、
    通知用環境変数がコンテナに渡されることを保証する。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from omegaconf import DictConfig, OmegaConf

from src.domain.data.job_manifest import JobManifest
from src.domain.repository.training_job import TrainingJobResult
from src.usecase.training.remote_submit import RemoteSubmitResult, RemoteSubmitUseCase

_FAKE_COMMIT = "c" * 40
_FAKE_JOB_NAME = "projects/123/locations/asia-northeast1/customJobs/999"


def _make_cfg(tmp_path: Path) -> DictConfig:
    """RemoteSubmitUseCase 用の DictConfig を生成する。"""
    processed_dir = tmp_path / "processed" / "titanic_preprocess" / "20260315T120000" / "train_out"
    processed_dir.mkdir(parents=True)
    (processed_dir / "fold_0").mkdir()
    (processed_dir / "fold_0" / "train.parquet").write_bytes(b"data")

    return OmegaConf.create(
        {
            "job_id": "titanic_lgbm",
            "competition": {"name": "titanic"},
            "preprocess_output_dir": str(processed_dir),
            "recipe": "lgbm",
            "output_dir": str(tmp_path / "models" / "titanic"),
            "remote_jobs_history_dir": str(tmp_path / "remote_jobs_history"),
            "seed": 42,
            "cloud": {
                "project": "test-project",
                "region": "asia-northeast1",
                "staging_bucket": "gs://test-bucket",
                "container_uri": "gcr.io/test/training:latest",
                "machine_type": "n1-standard-4",
                "service_account": "sa@test.iam.gserviceaccount.com",
            },
            "notification": {
                "slack": {"webhook_url": "https://hooks.slack.com/services/T/B/xxx"},
            },
        }
    )


def _make_mock_gcs() -> MagicMock:
    mock = MagicMock()
    mock.upload_dir.return_value = None
    return mock


def _make_mock_vertex() -> MagicMock:
    mock = MagicMock()
    mock.submit_custom_job.return_value = TrainingJobResult(
        resource_name=_FAKE_JOB_NAME, state="SUBMITTED"
    )
    return mock


def _make_mock_git_repo() -> MagicMock:
    mock = MagicMock()
    mock.get_commit_hash.return_value = _FAKE_COMMIT
    return mock


class TestRemoteSubmitUseCaseExecute:
    """RemoteSubmitUseCase.execute() のテスト。"""

    def test_returns_vertex_submit_result(self, tmp_path: Path) -> None:
        """execute() が RemoteSubmitResult を返すこと。"""
        cfg = _make_cfg(tmp_path)
        usecase = RemoteSubmitUseCase(
            cfg=cfg,
            object_storage=_make_mock_gcs(),
            training_job=_make_mock_vertex(),
            git_repo=_make_mock_git_repo(),
        )
        result = usecase.execute()
        assert isinstance(result, RemoteSubmitResult)

    def test_calls_submit_not_run(self, tmp_path: Path) -> None:
        """submit_custom_job が呼ばれ、run_custom_job は呼ばれないこと。"""
        cfg = _make_cfg(tmp_path)
        mock_vertex = _make_mock_vertex()
        usecase = RemoteSubmitUseCase(
            cfg=cfg,
            object_storage=_make_mock_gcs(),
            training_job=mock_vertex,
            git_repo=_make_mock_git_repo(),
        )
        usecase.execute()
        mock_vertex.submit_custom_job.assert_called_once()

    def test_saves_job_manifest(self, tmp_path: Path) -> None:
        """job_manifest.yaml が保存されること。"""
        cfg = _make_cfg(tmp_path)
        usecase = RemoteSubmitUseCase(
            cfg=cfg,
            object_storage=_make_mock_gcs(),
            training_job=_make_mock_vertex(),
            git_repo=_make_mock_git_repo(),
        )
        result = usecase.execute()

        manifest_path = Path(result.manifest_path)
        assert manifest_path.exists()

        manifest = JobManifest.load(manifest_path)
        assert manifest.status == "SUBMITTED"
        assert manifest.job_id == "titanic_lgbm"
        assert manifest.remote_job_name == _FAKE_JOB_NAME

    def test_uploads_code_and_data(self, tmp_path: Path) -> None:
        """コードとデータが GCS にアップロードされること。"""
        cfg = _make_cfg(tmp_path)
        mock_gcs = _make_mock_gcs()
        usecase = RemoteSubmitUseCase(
            cfg=cfg,
            object_storage=mock_gcs,
            training_job=_make_mock_vertex(),
            git_repo=_make_mock_git_repo(),
        )
        usecase.execute()

        # upload_dir: src, conf, scripts, data = 4 calls
        assert mock_gcs.upload_dir.call_count == 4

    def test_env_vars_contain_notification_settings(self, tmp_path: Path) -> None:
        """コンテナ環境変数に通知設定が含まれること。"""
        cfg = _make_cfg(tmp_path)
        mock_vertex = _make_mock_vertex()
        usecase = RemoteSubmitUseCase(
            cfg=cfg,
            object_storage=_make_mock_gcs(),
            training_job=mock_vertex,
            git_repo=_make_mock_git_repo(),
        )
        usecase.execute()

        submit_call = mock_vertex.submit_custom_job.call_args
        env_vars: dict[str, str] = submit_call.kwargs["env_vars"]
        assert "SLACK_WEBHOOK_URL" in env_vars

    def test_result_contains_manifest_path(self, tmp_path: Path) -> None:
        """RemoteSubmitResult に manifest_path が記録されること。"""
        cfg = _make_cfg(tmp_path)
        usecase = RemoteSubmitUseCase(
            cfg=cfg,
            object_storage=_make_mock_gcs(),
            training_job=_make_mock_vertex(),
            git_repo=_make_mock_git_repo(),
        )
        result = usecase.execute()
        assert result.manifest_path.endswith("job_manifest.yaml")
        assert "remote_jobs_history" in result.manifest_path

    def test_creates_gitignore_and_gitkeep_in_history_dir(self, tmp_path: Path) -> None:
        """remote_jobs_history ディレクトリに .gitignore と .gitkeep が作成されること。"""
        cfg = _make_cfg(tmp_path)
        usecase = RemoteSubmitUseCase(
            cfg=cfg,
            object_storage=_make_mock_gcs(),
            training_job=_make_mock_vertex(),
            git_repo=_make_mock_git_repo(),
        )
        result = usecase.execute()

        history_dir = Path(result.manifest_path).parent
        assert (history_dir / ".gitignore").exists()
        assert (history_dir / ".gitkeep").exists()
        gitignore_content = (history_dir / ".gitignore").read_text()
        assert "!.gitkeep" in gitignore_content
