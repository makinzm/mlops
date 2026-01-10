from dataclasses import dataclass
from pathlib import Path

from src.ports import ServingGateway


@dataclass
class SubmitInput:
    """提出の入力"""

    predictions_path: Path  # 予測結果ファイル
    submission_name: str  # 提出名


@dataclass
class SubmitOutput:
    """提出の出力"""

    submission_id: str  # 提出ID（Kaggleならsubmission ID）
    status: dict  # ステータス情報


class SubmitUseCase:
    """提出ユースケース

    I/F:
        Input: predictions_path, submission_name
        Output: submission_id, status

    責務:
        - 予測結果の読み込み
        - フォーマット変換（必要に応じて）
        - 提出実行（Kaggle API、API endpoint等）
    """

    def __init__(self, serving_gateway: ServingGateway) -> None:
        self.serving_gateway = serving_gateway

    def execute(self, input: SubmitInput) -> SubmitOutput:
        # TODO: プロジェクトごとに実装
        # predictions_df = pd.read_csv(input.predictions_path)
        # submission_id = self.serving_gateway.submit(predictions_df, input.submission_name)
        # status = self.serving_gateway.get_submission_status(submission_id)

        raise NotImplementedError("Implement submission for your project")
