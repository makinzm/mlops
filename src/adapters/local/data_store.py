from pathlib import Path

import pandas as pd


class LocalDataStore:
    """ローカルファイルシステム用DataStore実装"""

    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path
        self.raw_path = base_path / "raw"
        self.processed_path = base_path / "processed"
        self.features_path = base_path / "features"

    def load_raw(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.raw_path / f"{name}.csv")

    def save_processed(self, df: pd.DataFrame, name: str) -> Path:
        path = self.processed_path / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    def load_processed(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.processed_path / f"{name}.csv")

    def save_features(self, df: pd.DataFrame, name: str) -> Path:
        path = self.features_path / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    def load_features(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.features_path / f"{name}.csv")
