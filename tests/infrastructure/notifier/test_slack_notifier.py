"""
SlackNotifier の単体テスト。

なぜこのテストが必要か:
  - SlackNotifier は urllib.request で Slack Webhook に HTTP POST を送信する。
  - テストでは urllib.request.urlopen をモックし、実際の HTTP 通信を防ぐ。
  - 送信されるペイロードが Slack の Incoming Webhook 形式に準拠していることを保証する。
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.domain.repository.notifier import NotificationPayload
from src.infrastructure.notifier.slack_notifier import SlackNotifier


def _make_payload(status: str = "SUCCEEDED") -> NotificationPayload:
    return NotificationPayload(
        title="Training Job Completed",
        message="titanic_lgbm completed! CV: 0.832",
        job_id="titanic_lgbm",
        status=status,
        extra={"cv_score": "0.832"},
    )


class TestSlackNotifierSend:
    """SlackNotifier.send() のテスト。"""

    def test_sends_post_request_to_webhook_url(self) -> None:
        """webhook URL に POST リクエストが送信されること。"""
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/services/T/B/xxx")
        payload = _make_payload()

        with patch("src.infrastructure.notifier.slack_notifier.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()
            notifier.send(payload)

        mock_urlopen.assert_called_once()
        request = mock_urlopen.call_args[0][0]
        assert request.full_url == "https://hooks.slack.com/services/T/B/xxx"
        assert request.get_method() == "POST"

    def test_sends_json_body_with_text_field(self) -> None:
        """送信ボディに JSON の text フィールドが含まれること。"""
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/services/T/B/xxx")
        payload = _make_payload()

        with patch("src.infrastructure.notifier.slack_notifier.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()
            notifier.send(payload)

        request = mock_urlopen.call_args[0][0]
        body = json.loads(request.data.decode("utf-8"))
        assert "text" in body
        assert "titanic_lgbm" in body["text"]

    def test_includes_content_type_header(self) -> None:
        """Content-Type: application/json ヘッダーが設定されること。"""
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/services/T/B/xxx")
        payload = _make_payload()

        with patch("src.infrastructure.notifier.slack_notifier.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()
            notifier.send(payload)

        request = mock_urlopen.call_args[0][0]
        assert request.get_header("Content-type") == "application/json"

    def test_raises_on_http_error(self) -> None:
        """HTTP エラー時に例外が伝播すること。"""
        from urllib.error import URLError

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/services/T/B/xxx")
        payload = _make_payload()

        with patch("src.infrastructure.notifier.slack_notifier.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = URLError("Connection refused")
            with pytest.raises(URLError):
                notifier.send(payload)
