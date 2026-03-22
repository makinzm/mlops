"""
VertexAIRepositoryImpl の単体テスト。

なぜこのテストが必要か:
  - VertexAIRepositoryImpl は google-cloud-aiplatform を使い Vertex AI と通信する。
  - テストでは aiplatform モジュールをモックし、実際の GCP 呼び出しを防ぐ。
  - submit_custom_job が正しいパラメータでジョブを送信すること、
    wait_for_job がジョブ完了後に VertexJobStatus を返すことを保証する。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.domain.repository.vertex_ai import VertexJobStatus
from src.infrastructure.gcp.vertex_ai import VertexAIRepositoryImpl


class TestVertexAIRepositoryImplSubmitJob:
    """submit_custom_job のテスト。"""

    def test_submit_returns_job_resource_name(self) -> None:
        """ジョブ送信後にリソース名が返されること。"""
        mock_job_instance = MagicMock()
        mock_job_instance.resource_name = "projects/123/locations/asia-northeast1/customJobs/456"

        with patch("src.infrastructure.gcp.vertex_ai.aiplatform") as mock_aiplatform:
            mock_aiplatform.CustomJob.return_value = mock_job_instance
            repo = VertexAIRepositoryImpl(project="test-project", region="asia-northeast1")
            job_name = repo.submit_custom_job(
                display_name="test-job",
                container_uri="gcr.io/test/training:latest",
                args=["usecase=train", "recipe=lgbm"],
                machine_type="n1-standard-4",
                env_vars={"GCS_DATA_URI": "gs://bucket/data"},
                service_account="sa@test.iam.gserviceaccount.com",
            )

        assert job_name == "projects/123/locations/asia-northeast1/customJobs/456"
        mock_job_instance.submit.assert_called_once()

    def test_submit_passes_env_vars_to_container(self) -> None:
        """環境変数がコンテナスペックに含まれること。"""
        mock_job_instance = MagicMock()
        mock_job_instance.resource_name = "projects/123/locations/x/customJobs/1"
        created_specs: list[dict] = []

        def capture_job(**kwargs: object) -> MagicMock:
            created_specs.append(dict(kwargs))
            return mock_job_instance

        with patch("src.infrastructure.gcp.vertex_ai.aiplatform") as mock_aiplatform:
            mock_aiplatform.CustomJob.side_effect = capture_job
            repo = VertexAIRepositoryImpl(project="test-project", region="asia-northeast1")
            repo.submit_custom_job(
                display_name="test-job",
                container_uri="gcr.io/test/training:latest",
                args=[],
                machine_type="n1-standard-4",
                env_vars={"KEY": "VALUE"},
                service_account="sa@test.iam.gserviceaccount.com",
            )

        spec = created_specs[0]
        worker_specs = spec["worker_pool_specs"]
        container_env = worker_specs[0]["container_spec"]["env"]
        env_names = [e["name"] for e in container_env]
        assert "KEY" in env_names


class TestVertexAIRepositoryImplWaitForJob:
    """wait_for_job のテスト。"""

    def test_wait_returns_succeeded_status(self) -> None:
        """ジョブ成功時に SUCCEEDED ステータスが返されること。"""
        mock_job_instance = MagicMock()
        mock_job_instance.state.name = "JOB_STATE_SUCCEEDED"

        with patch("src.infrastructure.gcp.vertex_ai.aiplatform") as mock_aiplatform:
            mock_aiplatform.CustomJob.get.return_value = mock_job_instance
            repo = VertexAIRepositoryImpl(project="test-project", region="asia-northeast1")
            status = repo.wait_for_job("projects/123/locations/x/customJobs/1")

        assert status.state == "SUCCEEDED"
        assert status.is_succeeded

    def test_wait_returns_failed_status_with_error(self) -> None:
        """ジョブ失敗時に FAILED ステータスとエラーメッセージが返されること。"""
        mock_job_instance = MagicMock()
        mock_job_instance.state.name = "JOB_STATE_FAILED"
        mock_job_instance.error.message = "OOM error"

        with patch("src.infrastructure.gcp.vertex_ai.aiplatform") as mock_aiplatform:
            mock_aiplatform.CustomJob.get.return_value = mock_job_instance
            repo = VertexAIRepositoryImpl(project="test-project", region="asia-northeast1")
            status = repo.wait_for_job("projects/123/locations/x/customJobs/1")

        assert status.state == "FAILED"
        assert not status.is_succeeded
        assert status.error_message == "OOM error"


class TestVertexAIRepositoryImplCancelJob:
    """cancel_job のテスト。"""

    def test_cancel_calls_cancel_on_job(self) -> None:
        """cancel_job が Vertex AI ジョブの cancel を呼ぶこと。"""
        mock_job_instance = MagicMock()

        with patch("src.infrastructure.gcp.vertex_ai.aiplatform") as mock_aiplatform:
            mock_aiplatform.CustomJob.get.return_value = mock_job_instance
            repo = VertexAIRepositoryImpl(project="test-project", region="asia-northeast1")
            repo.cancel_job("projects/123/locations/x/customJobs/1")

        mock_job_instance.cancel.assert_called_once()


class TestVertexAIRepositoryImplListRunningJobs:
    """list_running_jobs のテスト。"""

    def test_returns_resource_names_of_running_jobs(self) -> None:
        """実行中ジョブのリソース名リストが返されること。"""
        mock_job1 = MagicMock()
        mock_job1.resource_name = "projects/123/locations/x/customJobs/1"
        mock_job2 = MagicMock()
        mock_job2.resource_name = "projects/123/locations/x/customJobs/2"

        with patch("src.infrastructure.gcp.vertex_ai.aiplatform") as mock_aiplatform:
            mock_aiplatform.CustomJob.list.return_value = [mock_job1, mock_job2]
            repo = VertexAIRepositoryImpl(project="test-project", region="asia-northeast1")
            job_names = repo.list_running_jobs()

        assert len(job_names) == 2
        assert "projects/123/locations/x/customJobs/1" in job_names
