from dataclasses import dataclass
from pathlib import Path

from src.ports import DataStore


@dataclass
class CreateDatasetInput:
    """データセット作成の入力"""

    raw_name: str  # 生データ名
    output_name: str  # 出力名


@dataclass
class CreateDatasetOutput:
    """データセット作成の出力"""

    path: Path
    num_samples: int
    num_features: int


class CreateDatasetUseCase:
    """データセット作成ユースケース

    I/F:
        Input: raw_name, output_name
        Output: path, num_samples, num_features

    責務:
        - 生データの読み込み
        - 特徴量エンジニアリング（core/features/を呼び出す）
        - 処理済みデータの保存
    """

    def __init__(self, data_store: DataStore) -> None:
        self.data_store = data_store

    def execute(self, input: CreateDatasetInput) -> CreateDatasetOutput:
        # TODO: プロジェクトごとに実装
        # raw_df = self.data_store.load_raw(input.raw_name)
        # features_df = create_features(raw_df)  # from core/features/
        # path = self.data_store.save_features(features_df, input.output_name)

        raise NotImplementedError("Implement dataset creation for your project")
