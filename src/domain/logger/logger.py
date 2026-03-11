"""
AppLogger Protocol。

GitRepository Protocol と同じパターンで、ログ出力の抽象を定義する。
UseCase は具体的なログ実装を知らず、このプロトコルにのみ依存する。
"""

from typing import Protocol


class AppLogger(Protocol):
    def info(self, message: str) -> None: ...

    def error(self, message: str, *, exc_info: bool = False) -> None: ...
