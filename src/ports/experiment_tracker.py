from typing import Any, Protocol


class ExperimentTracker(Protocol):
    """実験追跡を抽象化するポート"""

    def start_run(self, run_name: str | None = None) -> None:
        """実験ランを開始する"""
        ...

    def log_params(self, params: dict[str, Any]) -> None:
        """パラメータを記録する"""
        ...

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """メトリクスを記録する"""
        ...

    def log_artifact(self, path: str) -> None:
        """アーティファクトを記録する"""
        ...

    def end_run(self) -> None:
        """実験ランを終了する"""
        ...
