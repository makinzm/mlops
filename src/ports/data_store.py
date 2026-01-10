from pathlib import Path
from typing import Protocol

import pandas as pd


class DataStore(Protocol):
    """データの保存・読込を抽象化するポート"""

    def load_raw(self, name: str) -> pd.DataFrame:
        """生データを読み込む"""
        ...

    def save_processed(self, df: pd.DataFrame, name: str) -> Path:
        """処理済みデータを保存する"""
        ...

    def load_processed(self, name: str) -> pd.DataFrame:
        """処理済みデータを読み込む"""
        ...

    def save_features(self, df: pd.DataFrame, name: str) -> Path:
        """特徴量を保存する"""
        ...

    def load_features(self, name: str) -> pd.DataFrame:
        """特徴量を読み込む"""
        ...
