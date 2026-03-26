"""
EmailNotifier の単体テスト。

なぜこのテストが必要か:
  - EmailNotifier は smtplib で SMTP サーバーにメールを送信する。
  - テストでは smtplib.SMTP をモックし、実際のメール送信を防ぐ。
  - メールの件名・本文・宛先が正しく設定されることを保証する。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.domain.repository.notifier import NotificationPayload
from src.infrastructure.notifier.email_notifier import EmailNotifier


def _make_payload() -> NotificationPayload:
    return NotificationPayload(
        title="Training Job Completed",
        message="titanic_lgbm completed! CV: 0.832",
        job_id="titanic_lgbm",
        status="SUCCEEDED",
    )


class TestEmailNotifierSend:
    """EmailNotifier.send() のテスト。"""

    def test_sends_email_via_smtp(self) -> None:
        """SMTP サーバーにメールが送信されること。"""
        notifier = EmailNotifier(
            smtp_host="smtp.example.com",
            smtp_port=587,
            sender="noreply@example.com",
            recipient="user@example.com",
            username="noreply@example.com",
            password="secret",
        )
        payload = _make_payload()

        with patch("src.infrastructure.notifier.email_notifier.smtplib.SMTP") as mock_smtp_class:
            mock_smtp = MagicMock()
            mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)
            notifier.send(payload)

        mock_smtp.sendmail.assert_called_once()
        call_args = mock_smtp.sendmail.call_args
        assert call_args[0][0] == "noreply@example.com"
        assert call_args[0][1] == "user@example.com"

    def test_email_contains_job_id_in_body(self) -> None:
        """メール本文に job_id が含まれること。"""
        notifier = EmailNotifier(
            smtp_host="smtp.example.com",
            smtp_port=587,
            sender="noreply@example.com",
            recipient="user@example.com",
            username="noreply@example.com",
            password="secret",
        )
        payload = _make_payload()

        with patch("src.infrastructure.notifier.email_notifier.smtplib.SMTP") as mock_smtp_class:
            mock_smtp = MagicMock()
            mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)
            notifier.send(payload)

        call_args = mock_smtp.sendmail.call_args
        email_body: str = call_args[0][2]
        assert "titanic_lgbm" in email_body

    def test_starttls_is_called(self) -> None:
        """STARTTLS が呼ばれること。"""
        notifier = EmailNotifier(
            smtp_host="smtp.example.com",
            smtp_port=587,
            sender="noreply@example.com",
            recipient="user@example.com",
            username="noreply@example.com",
            password="secret",
        )
        payload = _make_payload()

        with patch("src.infrastructure.notifier.email_notifier.smtplib.SMTP") as mock_smtp_class:
            mock_smtp = MagicMock()
            mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)
            notifier.send(payload)

        mock_smtp.starttls.assert_called_once()

    def test_raises_on_smtp_error(self) -> None:
        """SMTP エラー時に例外が伝播すること。"""
        from smtplib import SMTPException

        notifier = EmailNotifier(
            smtp_host="smtp.example.com",
            smtp_port=587,
            sender="noreply@example.com",
            recipient="user@example.com",
            username="noreply@example.com",
            password="secret",
        )
        payload = _make_payload()

        with patch("src.infrastructure.notifier.email_notifier.smtplib.SMTP") as mock_smtp_class:
            mock_smtp_class.side_effect = SMTPException("Connection refused")
            with pytest.raises(SMTPException):
                notifier.send(payload)
