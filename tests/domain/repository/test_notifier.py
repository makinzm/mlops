"""
Notifier Protocol と NotificationPayload の単体テスト。

なぜこのテストが必要か:
  - Notifier Protocol のインターフェースが正しく定義されていることを runtime_checkable で保証する。
  - NotificationPayload の dataclass フィールドが期待通りに動作することを保証する。
"""

from __future__ import annotations

from src.domain.repository.notifier import NotificationPayload, Notifier


class _FakeNotifier:
    """Protocol 準拠テスト用のフェイク実装。"""

    def __init__(self) -> None:
        self.sent: list[NotificationPayload] = []

    def send(self, payload: NotificationPayload) -> None:
        self.sent.append(payload)


class TestNotificationPayload:
    """NotificationPayload のテスト。"""

    def test_creates_payload_with_required_fields(self) -> None:
        """必須フィールドのみで生成できること。"""
        payload = NotificationPayload(
            title="Job Done",
            message="Training completed successfully",
            job_id="titanic_lgbm",
            status="SUCCEEDED",
        )
        assert payload.title == "Job Done"
        assert payload.message == "Training completed successfully"
        assert payload.job_id == "titanic_lgbm"
        assert payload.status == "SUCCEEDED"
        assert payload.extra is None

    def test_creates_payload_with_extra(self) -> None:
        """extra フィールド付きで生成できること。"""
        payload = NotificationPayload(
            title="Job Done",
            message="Training completed",
            job_id="titanic_lgbm",
            status="SUCCEEDED",
            extra={
                "cv_score": "0.832",
                "download_command": "uv run python -m src usecase=vertex_download",
            },
        )
        assert payload.extra is not None
        assert payload.extra["cv_score"] == "0.832"


class TestNotifierProtocol:
    """Notifier Protocol のテスト。"""

    def test_fake_notifier_satisfies_protocol(self) -> None:
        """_FakeNotifier が Notifier Protocol を満たすこと（runtime_checkable）。"""
        notifier = _FakeNotifier()
        assert isinstance(notifier, Notifier)

    def test_fake_notifier_records_sent_payloads(self) -> None:
        """send() がペイロードを記録すること。"""
        notifier = _FakeNotifier()
        payload = NotificationPayload(
            title="Test",
            message="Test message",
            job_id="test_job",
            status="SUCCEEDED",
        )
        notifier.send(payload)
        assert len(notifier.sent) == 1
        assert notifier.sent[0].job_id == "test_job"
