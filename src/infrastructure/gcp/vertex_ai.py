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

        aiplatform_v1.JobServiceClient.create_custom_job() を直接使用する。
        高レベル API の job.run() はバックグラウンドスレッドで完了ポーリングを
        開始するため、プロセスが終了しない。低レベル API なら送信のみで即座に返る。

        ref: https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform_v1.services.job_service.JobServiceClient
        """
        from google.cloud import aiplatform_v1

        container_spec: dict[str, object] = {
            "image_uri": container_uri,
            "env": [{"name": k, "value": v} for k, v in env_vars.items()],
        }
        if command:
            container_spec["command"] = command
        if args:
            container_spec["args"] = args

        custom_job = {
            "display_name": display_name,
            "job_spec": {
                "worker_pool_specs": [
                    {
                        "machine_spec": {"machine_type": machine_type},
                        "replica_count": 1,
                        "container_spec": container_spec,
                    }
                ],
                "service_account": service_account,
            },
        }

        client = aiplatform_v1.JobServiceClient(
            client_options={"api_endpoint": f"{self._region}-aiplatform.googleapis.com"}
        )
        parent = f"projects/{self._project}/locations/{self._region}"

        logger.info(f"Submitting Vertex AI job: {display_name}")
        response = client.create_custom_job(request={"parent": parent, "custom_job": custom_job})

        resource_name: str = response.name
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

    def build_bootstrap_command(self, code_uri: str) -> list[str]:
        """GCS からコードをダウンロードし、依存をインストールしてエントリーポイントを実行する
        ブートストラップコマンドを生成する。

        Args:
            code_uri: コードが格納された GCS URI（例: gs://bucket/jobs/.../code）

        Returns:
            ["bash", "-c", "<bootstrap script>"]

        時間計算量: O(1)
        空間計算量: O(1)
        """
        gcs_download_py = (
            "import os; from google.cloud import storage; "
            f"uri='{code_uri}'; "
            "bkt,pfx=uri[5:].split('/',1); "
            "c=storage.Client(); "
            "[("
            "  os.makedirs(os.path.dirname(f'/app/{b.name[len(pfx)+1:]}'),exist_ok=True),"
            "  b.download_to_filename(f'/app/{b.name[len(pfx)+1:]}')"
            ") for b in c.list_blobs(bkt,prefix=pfx) if b.name[len(pfx)+1:]]"
        )
        bootstrap = (
            f'python -c "{gcs_download_py}"'
            " && pip install -q hydra-core omegaconf python-dotenv pydantic jinja2 mlflow"
            " torch torchvision albumentations grad-cam Pillow lightgbm polars"
            " && python /app/scripts/remote_entrypoint.py"
        )
        return ["bash", "-c", bootstrap]

    def build_console_url(self, job_name: str) -> str:
        """Vertex AI ジョブの Cloud Console URL を生成する。

        Args:
            job_name: projects/{number}/locations/{region}/customJobs/{id}

        Returns:
            Cloud Console URL

        時間計算量: O(1)
        空間計算量: O(1)
        """
        parts = job_name.split("/")
        return (
            f"https://console.cloud.google.com/vertex-ai/training/custom-jobs"
            f"/{parts[-1]}"
            f"?project={parts[1]}&region={parts[3]}"
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
