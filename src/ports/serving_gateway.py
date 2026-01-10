from pathlib import Path
from typing import Protocol

import pandas as pd


class ServingGateway(Protocol):
    """モデル提供を抽象化するポート（Kaggle提出・API公開・バッチ推論等）"""

    def submit(self, predictions: pd.DataFrame, name: str) -> str:
        """予測結果を提出/デプロイする。識別子を返す"""
        ...

    def get_submission_status(self, submission_id: str) -> dict:
        """提出/デプロイのステータスを取得する"""
        ...

    def export_for_serving(self, model_path: Path) -> Path:
        """モデルをサービング用にエクスポートする"""
        ...
