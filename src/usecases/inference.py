from dataclasses import dataclass
from pathlib import Path

from src.ports import DataStore, ModelStore


@dataclass
class InferenceInput:
    """推論の入力"""

    model_name: str  # 推論に使うモデル名
    features_name: str  # 推論対象データ名
    output_name: str  # 出力ファイル名


@dataclass
class InferenceOutput:
    """推論の出力"""

    predictions_path: Path
    num_predictions: int


class InferenceUseCase:
    """推論ユースケース

    I/F:
        Input: model_name, features_name, output_name
        Output: predictions_path, num_predictions

    責務:
        - モデルの読み込み
        - 推論対象データの読み込み
        - 予測実行
        - 結果の保存
    """

    def __init__(
        self,
        data_store: DataStore,
        model_store: ModelStore,
    ) -> None:
        self.data_store = data_store
        self.model_store = model_store

    def execute(self, input: InferenceInput) -> InferenceOutput:
        # TODO: プロジェクトごとに実装
        # model = self.model_store.load(input.model_name)
        # features_df = self.data_store.load_features(input.features_name)
        # predictions = model.predict(features_df)
        # path = save_predictions(predictions, input.output_name)

        raise NotImplementedError("Implement inference for your project")
