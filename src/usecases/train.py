from dataclasses import dataclass
from typing import Any

from src.ports import DataStore, ExperimentTracker, ModelStore


@dataclass
class TrainResult:
    model: Any
    metrics: dict[str, float]


class TrainUseCase:
    """モデル学習ユースケース"""

    def __init__(
        self,
        data_store: DataStore,
        model_store: ModelStore,
        tracker: ExperimentTracker,
    ) -> None:
        self.data_store = data_store
        self.model_store = model_store
        self.tracker = tracker

    def execute(self, config: dict[str, Any]) -> TrainResult:
        """学習を実行する（具体的な実装はプロジェクトごとに拡張）"""
        self.tracker.start_run(config.get("run_name"))
        self.tracker.log_params(config)

        # TODO: 具体的な学習ロジックをプロジェクトで実装
        # train_df = self.data_store.load_features("train")
        # model = ...
        # metrics = ...

        raise NotImplementedError("Implement training logic for your project")
