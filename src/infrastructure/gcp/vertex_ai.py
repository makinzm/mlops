"""
VertexAIRepositoryImpl — google-cloud-aiplatform を使った VertexAIRepository の実装。

run_custom_job: CustomJob を Vertex AI に送信し、完了まで待機する。
cancel_job: 実行中のジョブをキャンセルする。
list_running_jobs: 実行中のジョブ一覧を返す。
"""

from __future__ import annotations

import logging

from google.cloud import aiplatform

from src.domain.repository.training_job import TrainingJobResult

logger = logging.getLogger(__name__)

# Vertex AI ジョブ状態 → TrainingJobResult.state のマッピング
_STATE_MAP: dict[str, str] = {
    "JOB_STATE_SUCCEEDED": "SUCCEEDED",
    "JOB_STATE_FAILED": "FAILED",
    "JOB_STATE_CANCELLED": "CANCELLED",
    "JOB_STATE_CANCELLING": "CANCELLED",
}


class VertexAIRepositoryImpl:
    """VertexAIRepository の google-cloud-aiplatform による実装。"""

    def __init__(self, project: str, region: str, staging_bucket: str) -> None:
        aiplatform.init(project=project, location=region, staging_bucket=staging_bucket)
        self._project = project
        self._region = region

    def run_custom_job(
        self,
        display_name: str,
        container_uri: str,
        command: list[str],
        args: list[str],
        machine_type: str,
        env_vars: dict[str, str],
        service_account: str,
    ) -> TrainingJobResult:
        """カスタムトレーニングジョブを送信し、完了まで待機して結果を返す。

        内部で run(sync=True) を使用する。sync=True は送信 + 完了待機を
        1メソッドで行い、ジョブが失敗した場合は RuntimeError を送出する。
        ref: https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.CustomJob
        """
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
        logger.info(f"Submitting Vertex AI job: {display_name}")

        # run(sync=True) はジョブ送信 + 完了待機をブロッキングで実行する。
        # ジョブ失敗時は RuntimeError を送出する。
        try:
            job.run(service_account=service_account, sync=True)
        except RuntimeError:
            # run() が失敗時に RuntimeError を投げるが、
            # 結果を VertexJobResult で返したいので catch する
            pass

        resource_name: str = job.resource_name
        raw_state: str = job.state.name
        state = _STATE_MAP.get(raw_state, raw_state)
        error_msg: str | None = None
        if state == "FAILED":
            try:
                error = getattr(job, "error", None)
                error_msg = str(error.message) if error is not None else raw_state
            except AttributeError:
                error_msg = raw_state

        logger.info(f"Job {resource_name} finished with state: {state}")
        return TrainingJobResult(
            resource_name=resource_name,
            state=state,
            error_message=error_msg,
        )

    def submit_custom_job(
        self,
        display_name: str,
        container_uri: str,
        command: list[str],
        args: list[str],
        machine_type: str,
        env_vars: dict[str, str],
        service_account: str,
    ) -> TrainingJobResult:
        """カスタムトレーニングジョブを送信し、即座に返す（完了を待たない）。

        run(sync=False) を使用し、ジョブ送信後に即座に戻る。
        state='SUBMITTED' の TrainingJobResult を返す。
        ref: https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.CustomJob
        """
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
        logger.info(f"Submitting Vertex AI job (async): {display_name}")

        # sync=False: ジョブ送信後に即座に返る（完了を待たない）
        job.run(service_account=service_account, sync=False)

        resource_name: str = job.resource_name
        logger.info(f"Job submitted: {resource_name}")
        return TrainingJobResult(
            resource_name=resource_name,
            state="SUBMITTED",
        )

    def get_job_status(self, job_name: str) -> TrainingJobResult:
        """ジョブの現在のステータスを取得する。

        ref: https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.CustomJob
        """
        job = aiplatform.CustomJob.get(job_name)
        raw_state: str = job.state.name
        state = _STATE_MAP.get(raw_state, raw_state)

        error_msg: str | None = None
        if state == "FAILED":
            try:
                error = getattr(job, "error", None)
                error_msg = str(error.message) if error is not None else raw_state
            except AttributeError:
                error_msg = raw_state

        logger.info(f"Job {job_name} status: {state}")
        return TrainingJobResult(
            resource_name=job_name,
            state=state,
            error_message=error_msg,
        )

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
