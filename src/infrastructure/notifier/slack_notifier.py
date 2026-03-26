"""
SlackNotifier — Slack Incoming Webhook に HTTP POST で通知を送信する。

urllib.request を使用し、外部ライブラリに依存しない。
remote_entrypoint.py からも軽量に呼び出せることを意図している。
"""

from __future__ import annotations

import json
import logging
from urllib.request import Request, urlopen

from src.domain.repository.notifier import NotificationPayload

logger = logging.getLogger(__name__)


class SlackNotifier:
    """Slack Incoming Webhook に通知を送信する。

    Args:
        webhook_url: Slack Incoming Webhook の URL。
    """

    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url

    def send(self, payload: NotificationPayload) -> None:
        """Slack に通知を送信する。

        時間計算量: O(1)（HTTP リクエスト 1 回）
        空間計算量: O(M) — M: メッセージ長

        Args:
            payload: 送信する通知のペイロード。

        Raises:
            URLError: HTTP 通信に失敗した場合。
        """
        text_parts = [
            f"*{payload.title}*",
            f"Job: `{payload.job_id}` | Status: `{payload.status}`",
            payload.message,
        ]
        if payload.extra:
            for key, value in payload.extra.items():
                text_parts.append(f"{key}: {value}")

        body = json.dumps({"text": "\n".join(text_parts)}).encode("utf-8")
        request = Request(
            self._webhook_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urlopen(request)
        logger.info(f"Slack notification sent for job: {payload.job_id}")
