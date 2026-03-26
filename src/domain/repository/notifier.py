"""
通知リポジトリのドメイン定義。

なぜここに定義するか:
  UseCase 層は通知の具体的な送信手段（Slack, Email 等）を知らない。
  Protocol を通じてのみ依存し、Infrastructure 層に実装を閉じる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class NotificationPayload:
    """通知のペイロード。

    Attributes:
        title: 通知タイトル（例: "Training Job Completed"）
        message: 通知本文
        job_id: ジョブ識別子
        status: ジョブ状態（SUCCEEDED / FAILED）
        extra: 追加情報（CV スコア、コマンド例など）
    """

    title: str
    message: str
    job_id: str
    status: str
    extra: dict[str, str] | None = None


@runtime_checkable
class Notifier(Protocol):
    """通知送信の抽象 Protocol。"""

    def send(self, payload: NotificationPayload) -> None:
        """通知を送信する。

        Args:
            payload: 送信する通知のペイロード。

        Raises:
            NotificationError: 送信に失敗した場合
        """
        ...
