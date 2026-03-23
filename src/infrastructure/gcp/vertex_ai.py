"""
VertexAIRepositoryImpl — google-cloud-aiplatform を使った VertexAIRepository の実装。

submit_custom_job: CustomJob を Vertex AI に送信する。
wait_for_job: ジョブ完了までポーリングし VertexJobStatus を返す。
cancel_job: 実行中のジョブをキャンセルする。
list_running_jobs: 実行中のジョブ一覧を返す（予算超過対策）。
"""

from __future__ import annotations

import logging

from google.cloud import aiplatform

from src.domain.repository.vertex_ai import VertexJobStatus

logger = logging.getLogger(__name__)

# Vertex AI ジョブ状態 → VertexJobStatus.state のマッピング
_STATE_MAP: dict[str, str] = {
    "JOB_STATE_SUCCEEDED": "SUCCEEDED",
    "JOB_STATE_FAILED": "FAILED",
    "JOB_STATE_CANCELLED": "CANCELLED",
    "JOB_STATE_CANCELLING": "CANCELLED",
}
_RUNNING_STATES: set[str] = {
    "JOB_STATE_RUNNING",
    "JOB_STATE_QUEUED",
    "JOB_STATE_PENDING",
}


class VertexAIRepositoryImpl:
    """VertexAIRepository の google-cloud-aiplatform による実装。"""

    def __init__(self, project: str, region: str, staging_bucket: str) -> None:
        aiplatform.init(project=project, location=region, staging_bucket=staging_bucket)
        self._project = project
        self._region = region
        self._jobs: dict[str, object] = {}  # resource_name → job instance

    def submit_custom_job(
        self,
        display_name: str,
        container_uri: str,
        command: list[str],
        args: list[str],
        machine_type: str,
        env_vars: dict[str, str],
        service_account: str,
    ) -> str:
        """カスタムトレーニングジョブを送信し、ジョブリソース名を返す。"""
        container_spec: dict[str, object] = {
            "image_uri": container_uri,
            "env": [{"name": k, "value": v} for k, v in env_vars.items()],
        }
        if command:
            container_spec["command"] = command
        if args:
            container_spec["args"] = args
        worker_pool_specs = [
            {
                "machine_spec": {"machine_type": machine_type},
                "replica_count": 1,
                "container_spec": container_spec,
            }
        ]
        job = aiplatform.CustomJob(
            display_name=display_name,
            worker_pool_specs=worker_pool_specs,
        )
        job.submit(service_account=service_account)
        resource_name: str = job.resource_name
        self._jobs[resource_name] = job
        logger.info(f"Submitted Vertex AI job: {resource_name}")
        return resource_name

    def wait_for_job(self, job_name: str) -> VertexJobStatus:
        """ジョブ完了までポーリングし、最終状態を返す。"""
        # submit() したインスタンスを使う（get() だと wait() が即座に返る場合がある）
        job = self._jobs.pop(job_name, None) or aiplatform.CustomJob.get(job_name)
        job.wait()  # type: ignore[union-attr,attr-defined]
        raw_state: str = job.state.name  # type: ignore[union-attr,attr-defined]
        state = _STATE_MAP.get(raw_state, raw_state)
        error_msg: str | None = None
        if state == "FAILED":
            try:
                error = getattr(job, "error", None)
                error_msg = str(error.message) if error is not None else raw_state
            except AttributeError:
                error_msg = raw_state
        logger.info(f"Job {job_name} finished with state: {state}")
        return VertexJobStatus(state=state, error_message=error_msg)

    def cancel_job(self, job_name: str) -> None:
        """実行中のジョブをキャンセルする。"""
        job = aiplatform.CustomJob.get(job_name)
        job.cancel()
        logger.info(f"Cancelled job: {job_name}")

    def list_running_jobs(self) -> list[str]:
        """実行中のジョブリソース名一覧を返す。"""
        jobs = aiplatform.CustomJob.list(
            filter='state="JOB_STATE_RUNNING" OR state="JOB_STATE_QUEUED"'
        )
        return [j.resource_name for j in jobs]
