"""
表形式データの抽象プロトコル。

Domain 層がインフラ（polars/pandas）に依存しないよう、
必要な操作のみを Protocol として定義する。
polars/pandas の実装はこのプロトコルを自然に満たす（duck typing）。
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Series(Protocol):
    """列データの抽象プロトコル。"""

    def to_list(self) -> list[object]: ...


@runtime_checkable
class DataFrame(Protocol):
    """表形式データの抽象プロトコル。

    polars.DataFrame および pandas.DataFrame はこのプロトコルを自然に満たす。
    インフラ層でのラッパーは不要。
    """

    columns: list[str]

    def __len__(self) -> int: ...
    def __getitem__(self, key: str) -> Series: ...
