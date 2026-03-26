"""
EmailNotifier — smtplib で SMTP サーバー経由のメール通知を送信する。

標準ライブラリのみ使用し、外部依存なし。
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.text import MIMEText

from src.domain.repository.notifier import NotificationPayload

logger = logging.getLogger(__name__)


class EmailNotifier:
    """SMTP サーバー経由でメール通知を送信する。

    Args:
        smtp_host: SMTP サーバーのホスト名。
        smtp_port: SMTP サーバーのポート番号。
        sender: 送信元メールアドレス。
        recipient: 送信先メールアドレス。
        username: SMTP 認証ユーザー名。
        password: SMTP 認証パスワード。
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        sender: str,
        recipient: str,
        username: str,
        password: str,
    ) -> None:
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._sender = sender
        self._recipient = recipient
        self._username = username
        self._password = password

    def send(self, payload: NotificationPayload) -> None:
        """メールで通知を送信する。

        時間計算量: O(1)（SMTP セッション 1 回）
        空間計算量: O(M) — M: メッセージ長

        Args:
            payload: 送信する通知のペイロード。

        Raises:
            SMTPException: SMTP 通信に失敗した場合。
        """
        body_parts = [
            f"Job: {payload.job_id}",
            f"Status: {payload.status}",
            "",
            payload.message,
        ]
        if payload.extra:
            body_parts.append("")
            for key, value in payload.extra.items():
                body_parts.append(f"{key}: {value}")

        msg = MIMEText("\n".join(body_parts))
        msg["Subject"] = payload.title
        msg["From"] = self._sender
        msg["To"] = self._recipient

        with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
            server.starttls()
            server.login(self._username, self._password)
            server.sendmail(self._sender, self._recipient, msg.as_string())

        logger.info(f"Email notification sent for job: {payload.job_id}")
