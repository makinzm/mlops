"""
Python 標準ライブラリを使った AppLogger 実装。
"""

import logging


class PythonAppLogger:
    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def info(self, message: str) -> None:
        self._logger.info(message)

    def error(self, message: str, *, exc_info: bool = False) -> None:
        self._logger.error(message, exc_info=exc_info)
