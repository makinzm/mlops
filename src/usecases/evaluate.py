from dataclasses import dataclass
from typing import Any

from src.ports import DataStore, ModelStore


@dataclass
class EvaluateInput:
    """評価の入力"""

    model_name: str  # 評価するモデル名
    features_name: str  # 評価用データ名


@dataclass
class EvaluateOutput:
    """評価の出力"""

    metrics: dict[str, float]  # {"rmse": 0.1, "mae": 0.05, ...}
    details: dict[str, Any]  # fold別スコア、混同行列など


class EvaluateUseCase:
    """モデル評価ユースケース（Check Model）

    I/F:
        Input: model_name, features_name
        Output: metrics, details

    責務:
        - 学習済みモデルの読み込み
        - 評価データでの推論
        - メトリクス計算（core/metrics/を使用）
        - 詳細レポート生成
    """

    def __init__(
        self,
        data_store: DataStore,
        model_store: ModelStore,
    ) -> None:
        self.data_store = data_store
        self.model_store = model_store

    def execute(self, input: EvaluateInput) -> EvaluateOutput:
        # TODO: プロジェクトごとに実装
        # model = self.model_store.load(input.model_name)
        # features_df = self.data_store.load_features(input.features_name)
        # predictions = model.predict(features_df)
        # metrics = calculate_metrics(features_df["target"], predictions)

        raise NotImplementedError("Implement evaluation for your project")
