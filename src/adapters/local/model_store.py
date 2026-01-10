import pickle
from pathlib import Path
from typing import Any


class LocalModelStore:
    """ローカルファイルシステム用ModelStore実装"""

    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, model: Any, name: str, metadata: dict[str, Any] | None = None) -> Path:
        path = self.base_path / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": model, "metadata": metadata}, f)
        return path

    def load(self, name: str) -> Any:
        path = self.base_path / f"{name}.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["model"]

    def list_models(self) -> list[str]:
        return [p.stem for p in self.base_path.glob("*.pkl")]
