"""
Budget Enforcer Cloud Function.

Pub/Sub から予算超過通知を受け取り、
BUDGET_ACTION=stop の場合は実行中の Vertex AI ジョブを全停止する。
BUDGET_ACTION=warn の場合はログ出力のみ。
"""

import base64
import json
import logging
import os

import functions_framework

logger = logging.getLogger(__name__)


@functions_framework.cloud_event
def handle_budget_alert(cloud_event):  # type: ignore[no-untyped-def]
    """Pub/Sub 予算アラートを処理する。"""
    budget_action = os.environ.get("BUDGET_ACTION", "warn")
    project = os.environ.get("GCP_PROJECT", "")
    region = os.environ.get("GCP_REGION", "asia-northeast1")

    # Pub/Sub メッセージをデコード
    try:
        data = base64.b64decode(cloud_event.data["message"]["data"]).decode("utf-8")
        budget_info = json.loads(data)
        budget_name = budget_info.get("budgetDisplayName", "unknown")
        cost_amount = budget_info.get("costAmount", 0)
        budget_amount = budget_info.get("budgetAmount", 0)
        logger.warning(
            "Budget alert: %s — cost=$%.2f / budget=$%.2f",
            budget_name,
            float(cost_amount),
            float(budget_amount),
        )
    except Exception as e:
        logger.error("Failed to parse budget alert: %s", e)
        return

    if budget_action != "stop":
        logger.info("budget_action=%s: logging only, no job cancellation", budget_action)
        return

    # budget_action = "stop": 実行中ジョブを全停止
    # aiplatform は重いため stop 時のみ lazy import
    from google.cloud import aiplatform  # type: ignore[import-untyped]

    logger.warning("budget_action=stop: cancelling all running Vertex AI jobs")
    aiplatform.init(project=project, location=region)
    running_jobs = aiplatform.CustomJob.list(
        filter='state="JOB_STATE_RUNNING" OR state="JOB_STATE_QUEUED"'
    )
    for job in running_jobs:
        try:
            job.cancel()
            logger.warning("Cancelled job: %s", job.resource_name)
        except Exception as e:
            logger.error("Failed to cancel job %s: %s", job.resource_name, e)
