from dataclasses import dataclass

import pandas as pd

from src.ports import ServingGateway


@dataclass
class SubmitResult:
    submission_id: str
    status: dict


class SubmitUseCase:
    """提出/デプロイユースケース"""

    def __init__(self, serving_gateway: ServingGateway) -> None:
        self.serving_gateway = serving_gateway

    def execute(self, predictions: pd.DataFrame, name: str) -> SubmitResult:
        """予測結果を提出/デプロイする"""
        submission_id = self.serving_gateway.submit(predictions, name)
        status = self.serving_gateway.get_submission_status(submission_id)
        return SubmitResult(submission_id=submission_id, status=status)
