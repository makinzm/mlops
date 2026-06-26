"""
JobDownloadUseCase の単体テスト。

なぜこのテストが必要か:
  - JobDownloadUseCase は manifest を読み込み、学習ジョブのステータスを確認し、
    成功していればオブジェクトストレージからモデルをダウンロードして manifest を更新する。
  - ジョブ未完了・失敗時の適切なエラーハンドリングを保証する。
  - ダウンロード後に manifest の status が DOWNLOADED に更新されることを保証する。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.domain.data.job_manifest import JobManifest
from src.domain.repository.training_job import TrainingJobResult
from src.usecase.training.job_download import JobDownloadResult, JobDownloadUseCase

_FAKE_JOB_NAME = "projects/123/locations/asia-northeast1/customJobs/999"


def _make_manifest(tmp_path: Path, status: str = "SUBMITTED") -> tuple[JobManifest, Path]:
    """テスト用の manifest を作成して保存する。"""
    manifest = JobManifest(
        job_id="titanic_lgbm",
        competition="titanic",
        recipe="lgbm",
        timestamp="20260325T143000",
        commit_hash="d" * 40,
        status=status,
        cloud_job_name=_FAKE_JOB_NAME,
        gcs_code_uri="gs://test-bucket/jobs/titanic_lgbm/20260325T143000/code",
        gcs_data_uri="gs://test-bucket/jobs/titanic_lgbm/20260325T143000/data",
        gcs_model_uri="gs://test-bucket/jobs/titanic_lgbm/20260325T143000/models",
        submitted_at="2026-03-25T14:30:00",
    )
    manifest_path = tmp_path / "models" / "titanic" / "titanic_lgbm" / "job_manifest.yaml"
    manifest.save(manifest_path)
    return manifest, manifest_path


def _make_mock_vertex(state: str = "SUCCEEDED") -> MagicMock:
    mock = MagicMock()
    mock.get_job_status.return_value = TrainingJobResult(
        resource_name=_FAKE_JOB_NAME,
        state=state,
        error_message="OOM error" if state == "FAILED" else None,
    )
    return mock


def _make_mock_gcs() -> MagicMock:
    mock = MagicMock()
    mock.download_dir.return_value = None
    return mock


class TestJobDownloadUseCaseExecute:
    """JobDownloadUseCase.execute() のテスト。"""

    def test_returns_download_result_on_success(self, tmp_path: Path) -> None:
        """ジョブ成功時に JobDownloadResult を返すこと。"""
        _, manifest_path = _make_manifest(tmp_path)
        usecase = JobDownloadUseCase(
            manifest_path=manifest_path,
            object_storage=_make_mock_gcs(),
            training_job=_make_mock_vertex("SUCCEEDED"),
            output_dir=tmp_path / "models" / "titanic",
        )
        result = usecase.execute()
        assert isinstance(result, JobDownloadResult)

    def test_downloads_model_from_gcs(self, tmp_path: Path) -> None:
        """GCS からモデルがダウンロードされること。"""
        _, manifest_path = _make_manifest(tmp_path)
        mock_gcs = _make_mock_gcs()
        usecase = JobDownloadUseCase(
            manifest_path=manifest_path,
            object_storage=mock_gcs,
            training_job=_make_mock_vertex("SUCCEEDED"),
            output_dir=tmp_path / "models" / "titanic",
        )
        usecase.execute()
        mock_gcs.download_dir.assert_called_once()

    def test_updates_manifest_to_downloaded(self, tmp_path: Path) -> None:
        """ダウンロード後に manifest の status が DOWNLOADED に更新されること。"""
        _, manifest_path = _make_manifest(tmp_path)
        usecase = JobDownloadUseCase(
            manifest_path=manifest_path,
            object_storage=_make_mock_gcs(),
            training_job=_make_mock_vertex("SUCCEEDED"),
            output_dir=tmp_path / "models" / "titanic",
        )
        usecase.execute()

        updated = JobManifest.load(manifest_path)
        assert updated.status == "DOWNLOADED"
        assert updated.local_model_dir is not None

    def test_raises_on_job_still_running(self, tmp_path: Path) -> None:
        """ジョブがまだ実行中の場合に RuntimeError が送出されること。"""
        _, manifest_path = _make_manifest(tmp_path)
        mock_vertex = MagicMock()
        mock_vertex.get_job_status.return_value = TrainingJobResult(
            resource_name=_FAKE_JOB_NAME, state="RUNNING"
        )
        usecase = JobDownloadUseCase(
            manifest_path=manifest_path,
            object_storage=_make_mock_gcs(),
            training_job=mock_vertex,
            output_dir=tmp_path / "models" / "titanic",
        )
        with pytest.raises(RuntimeError, match="RUNNING"):
            usecase.execute()

    def test_raises_on_job_failed(self, tmp_path: Path) -> None:
        """ジョブが FAILED の場合に RuntimeError が送出されること。"""
        _, manifest_path = _make_manifest(tmp_path)
        usecase = JobDownloadUseCase(
            manifest_path=manifest_path,
            object_storage=_make_mock_gcs(),
            training_job=_make_mock_vertex("FAILED"),
            output_dir=tmp_path / "models" / "titanic",
        )
        with pytest.raises(RuntimeError, match="FAILED"):
            usecase.execute()

    def test_checks_job_status_via_training_job(self, tmp_path: Path) -> None:
        """get_job_status が cloud_job_name で呼ばれること。"""
        _, manifest_path = _make_manifest(tmp_path)
        mock_vertex = _make_mock_vertex("SUCCEEDED")
        usecase = JobDownloadUseCase(
            manifest_path=manifest_path,
            object_storage=_make_mock_gcs(),
            training_job=mock_vertex,
            output_dir=tmp_path / "models" / "titanic",
        )
        usecase.execute()
        mock_vertex.get_job_status.assert_called_once_with(_FAKE_JOB_NAME)

    def test_result_contains_local_model_dir(self, tmp_path: Path) -> None:
        """JobDownloadResult に local_model_dir が記録されること。"""
        _, manifest_path = _make_manifest(tmp_path)
        usecase = JobDownloadUseCase(
            manifest_path=manifest_path,
            object_storage=_make_mock_gcs(),
            training_job=_make_mock_vertex("SUCCEEDED"),
            output_dir=tmp_path / "models" / "titanic",
        )
        result = usecase.execute()
        assert result.local_model_dir is not None

    def test_skips_download_if_already_downloaded(self, tmp_path: Path) -> None:
        """manifest が既に DOWNLOADED の場合はダウンロードをスキップすること。"""
        _, manifest_path = _make_manifest(tmp_path, status="DOWNLOADED")
        mock_gcs = _make_mock_gcs()
        usecase = JobDownloadUseCase(
            manifest_path=manifest_path,
            object_storage=mock_gcs,
            training_job=_make_mock_vertex("SUCCEEDED"),
            output_dir=tmp_path / "models" / "titanic",
        )
        result = usecase.execute()
        mock_gcs.download_dir.assert_not_called()
        assert isinstance(result, JobDownloadResult)
