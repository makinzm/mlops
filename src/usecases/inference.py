from dataclasses import dataclass

import pandas as pd

from src.ports import DataStore, ModelStore


@dataclass
class InferenceResult:
    predictions: pd.DataFrame


class InferenceUseCase:
    """推論ユースケース"""

    def __init__(
        self,
        data_store: DataStore,
        model_store: ModelStore,
    ) -> None:
        self.data_store = data_store
        self.model_store = model_store

    def execute(self, model_name: str, data_name: str) -> InferenceResult:
        """推論を実行する（具体的な実装はプロジェクトごとに拡張）"""
        # TODO: 具体的な推論ロジックをプロジェクトで実装
        # model = self.model_store.load(model_name)
        # data = self.data_store.load_features(data_name)
        # predictions = model.predict(data)

        raise NotImplementedError("Implement inference logic for your project")
