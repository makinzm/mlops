"""
StepResolver Protocol。

全 Resolver が実装すべきインターフェースを定義する。
UseCase / DAGRunner は具体的な Resolver クラスを知らず、この Protocol に依存する。
"""

from typing import Protocol

import polars as pl


class StepResolver(Protocol):
    """前処理ステップを解決するための Protocol。

    各 Resolver は自分が対応するメソッド名のセットを supported_methods() で返す。
    DAGRunner は supported_methods() でメソッド存在確認を行い、
    未対応の場合は StepResult(status="skipped", reason="method not found in resolver") を返す。
    """

    def supported_methods(self) -> set[str]:
        """このResolver が対応するメソッド名のセットを返す。"""
        ...

    def execute(self, df: pl.DataFrame, method: str, **kwargs: object) -> pl.DataFrame:
        """指定メソッドを実行して変換後の DataFrame を返す。"""
        ...
