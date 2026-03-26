"""
CompositeNotifier — 複数の Notifier を順次呼び出す。

1つの notifier が失敗しても残りの notifier は実行される（部分失敗許容）。
"""

from __future__ import annotations

import logging

from src.domain.repository.notifier import NotificationPayload, Notifier

logger = logging.getLogger(__name__)


class CompositeNotifier:
    """複数の Notifier を順次呼び出す Composite パターンの実装。

    Args:
        notifiers: 呼び出す Notifier のリスト。
    """

    def __init__(self, notifiers: list[Notifier]) -> None:
        self._notifiers = notifiers

    def send(self, payload: NotificationPayload) -> None:
        """全 notifier に通知を送信する。部分失敗を許容する。

        時間計算量: O(N) — N: notifier 数
        空間計算量: O(1)

        Args:
            payload: 送信する通知のペイロード。
        """
        for notifier in self._notifiers:
            try:
                notifier.send(payload)
            except Exception:
                logger.warning(
                    f"Notifier {type(notifier).__name__} failed for job {payload.job_id}",
                    exc_info=True,
                )
