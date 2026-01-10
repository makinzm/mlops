from pathlib import Path

import pandas as pd

from src.adapters.kaggle.config import KaggleConfig


class KaggleDataStore:
    """Kaggle environment DataStore implementation

    Designed to work within Kaggle Notebooks where:
    - Input data is at /kaggle/input/{competition}
    - Output goes to /kaggle/working
    """

    def __init__(self, config: KaggleConfig) -> None:
        self.config = config
        self.input_path = config.competition_data_path
        self.output_path = config.output_path

    def load_raw(self, name: str) -> pd.DataFrame:
        path = self.input_path / f"{name}.csv"
        return pd.read_csv(path)

    def save_processed(self, df: pd.DataFrame, name: str) -> Path:
        path = self.output_path / "processed" / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    def load_processed(self, name: str) -> pd.DataFrame:
        path = self.output_path / "processed" / f"{name}.csv"
        return pd.read_csv(path)

    def save_features(self, df: pd.DataFrame, name: str) -> Path:
        path = self.output_path / "features" / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    def load_features(self, name: str) -> pd.DataFrame:
        path = self.output_path / "features" / f"{name}.csv"
        return pd.read_csv(path)
