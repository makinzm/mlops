from pathlib import Path
from typing import Any, Protocol


class ModelStore(Protocol):
    """モデルの保存・読込を抽象化するポート"""

    def save(self, model: Any, name: str, metadata: dict[str, Any] | None = None) -> Path:
        """モデルを保存する"""
        ...

    def load(self, name: str) -> Any:
        """モデルを読み込む"""
        ...

    def list_models(self) -> list[str]:
        """保存されているモデル一覧を取得する"""
        ...
