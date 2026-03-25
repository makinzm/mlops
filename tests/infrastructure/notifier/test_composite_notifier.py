"""
CompositeNotifier の単体テスト。

なぜこのテストが必要か:
  - CompositeNotifier は複数の Notifier を順次呼び出す。
  - 1つの notifier が失敗しても残りの notifier が実行されることを保証する（部分失敗許容）。
  - 全 notifier が呼び出されることを保証する。
"""

from __future__ import annotations

from src.domain.repository.notifier import NotificationPayload
from src.infrastructure.notifier.composite_notifier import CompositeNotifier


def _make_payload() -> NotificationPayload:
    return NotificationPayload(
        title="Test",
        message="Test message",
        job_id="test_job",
        status="SUCCEEDED",
    )


class _FakeNotifier:
    """テスト用のフェイク Notifier。"""

    def __init__(self, should_fail: bool = False) -> None:
        self.sent: list[NotificationPayload] = []
        self._should_fail = should_fail

    def send(self, payload: NotificationPayload) -> None:
        if self._should_fail:
            raise RuntimeError("Fake notifier error")
        self.sent.append(payload)


class TestCompositeNotifierSend:
    """CompositeNotifier.send() のテスト。"""

    def test_calls_all_notifiers(self) -> None:
        """全 notifier が呼び出されること。"""
        notifier_a = _FakeNotifier()
        notifier_b = _FakeNotifier()
        composite = CompositeNotifier(notifiers=[notifier_a, notifier_b])
        composite.send(_make_payload())

        assert len(notifier_a.sent) == 1
        assert len(notifier_b.sent) == 1

    def test_continues_on_partial_failure(self) -> None:
        """1つの notifier が失敗しても残りが実行されること。"""
        failing = _FakeNotifier(should_fail=True)
        succeeding = _FakeNotifier()
        composite = CompositeNotifier(notifiers=[failing, succeeding])
        composite.send(_make_payload())

        assert len(succeeding.sent) == 1

    def test_does_not_raise_on_failure(self) -> None:
        """notifier 失敗時でも例外が伝播しないこと。"""
        failing = _FakeNotifier(should_fail=True)
        composite = CompositeNotifier(notifiers=[failing])
        # CompositeNotifier は失敗しても例外を投げない
        composite.send(_make_payload())

    def test_empty_notifiers_does_nothing(self) -> None:
        """空の notifier リストで send() がエラーなく実行されること。"""
        composite = CompositeNotifier(notifiers=[])
        composite.send(_make_payload())
