from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.ports import DataStore, ExperimentTracker, ModelStore


@dataclass
class TrainInput:
    """学習の入力"""

    features_name: str  # 特徴量データ名
    model_name: str  # 保存するモデル名
    params: dict[str, Any]  # ハイパーパラメータ


@dataclass
class TrainOutput:
    """学習の出力"""

    model_path: Path
    metrics: dict[str, float]  # {"loss": 0.1, "accuracy": 0.95, ...}
    run_id: str | None  # 実験トラッキングのrun ID


class TrainUseCase:
    """モデル学習ユースケース

    I/F:
        Input: features_name, model_name, params
        Output: model_path, metrics, run_id

    責務:
        - 特徴量データの読み込み
        - モデルの学習（core/models/を使用）
        - メトリクスの記録
        - モデルの保存
    """

    def __init__(
        self,
        data_store: DataStore,
        model_store: ModelStore,
        tracker: ExperimentTracker,
    ) -> None:
        self.data_store = data_store
        self.model_store = model_store
        self.tracker = tracker

    def execute(self, input: TrainInput) -> TrainOutput:
        # TODO: プロジェクトごとに実装
        # self.tracker.start_run(input.model_name)
        # self.tracker.log_params(input.params)
        # features_df = self.data_store.load_features(input.features_name)
        # model = MyModel(**input.params)  # from core/models/
        # model.fit(features_df)
        # metrics = {"loss": ..., "accuracy": ...}
        # self.tracker.log_metrics(metrics)
        # path = self.model_store.save(model, input.model_name)
        # self.tracker.end_run()

        raise NotImplementedError("Implement training for your project")
