"""
リモート学習ジョブリポジトリのドメイン定義。

なぜここに定義するか:
  UseCase 層はリモート学習基盤の具体 SDK を知らない。Protocol を通じてのみ依存する。
  これにより特定クラウド依存を infrastructure 層に閉じられる。
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class TrainingJobResult:
    """リモート学習ジョブの実行結果。"""

    resource_name: str
    state: str  # "SUCCEEDED" | "FAILED" | "CANCELLED"
    error_message: str | None = None

    @property
    def is_succeeded(self) -> bool:
        return self.state == "SUCCEEDED"


@runtime_checkable
class TrainingJobRepository(Protocol):
    """リモート学習ジョブ操作の抽象 Protocol。"""

    def run_custom_job(
        self,
        display_name: str,
        container_uri: str,
        command: list[str],
        args: list[str],
        machine_type: str,
        env_vars: dict[str, str],
        service_account: str,
    ) -> TrainingJobResult:
        """カスタムトレーニングジョブを送信し、完了まで待機して結果を返す。"""
        ...

    def submit_custom_job(
        self,
        display_name: str,
        container_uri: str,
        command: list[str],
        args: list[str],
        machine_type: str,
        env_vars: dict[str, str],
        service_account: str,
    ) -> TrainingJobResult:
        """カスタムトレーニングジョブを送信し、即座に返す（完了を待たない）。"""
        ...

    def get_job_status(self, job_name: str) -> TrainingJobResult:
        """ジョブの現在のステータスを取得する。"""
        ...

    def cancel_job(self, job_name: str) -> None:
        """実行中のジョブをキャンセルする。"""
        ...

    def list_running_jobs(self) -> list[str]:
        """実行中のジョブリソース名一覧を返す。"""
        ...
