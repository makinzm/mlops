from dataclasses import dataclass
from pathlib import Path

from src.ports import ModelStore, ServingGateway


@dataclass
class ServeInput:
    """サービング準備の入力"""

    model_name: str  # サービングするモデル名


@dataclass
class ServeOutput:
    """サービング準備の出力"""

    artifact_path: Path  # エクスポートされたアーティファクト
    serving_info: dict  # エンドポイント情報など


class ServeUseCase:
    """モデルサービング準備ユースケース

    I/F:
        Input: model_name
        Output: artifact_path, serving_info

    責務:
        - モデルのエクスポート（ONNX変換など）
        - サービング環境への配置準備
    """

    def __init__(
        self,
        model_store: ModelStore,
        serving_gateway: ServingGateway,
    ) -> None:
        self.model_store = model_store
        self.serving_gateway = serving_gateway

    def execute(self, input: ServeInput) -> ServeOutput:
        # TODO: プロジェクトごとに実装
        # model_path = self.model_store.get_path(input.model_name)
        # artifact_path = self.serving_gateway.export_for_serving(model_path)

        raise NotImplementedError("Implement serving for your project")
