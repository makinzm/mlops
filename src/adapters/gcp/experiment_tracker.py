from typing import Any

import mlflow
from mlflow.entities import Run

from src.adapters.gcp.config import GCPConfig


class VertexMLflowTracker:
    """Vertex AI MLflow integration for ExperimentTracker port

    Uses Vertex AI's MLflow integration for experiment tracking.
    MLflow UI is accessed via Vertex AI Experiments.
    """

    def __init__(self, config: GCPConfig, experiment_name: str = "default") -> None:
        self.config = config
        self.experiment_name = experiment_name

        # Set MLflow tracking URI
        # For Vertex AI: use GCS-backed tracking or Vertex AI Experiments
        tracking_uri = config.mlflow_tracking_uri or f"gs://{config.bucket}/mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self._run: Run | None = None

    def start_run(self, run_name: str | None = None) -> None:
        self._run = mlflow.start_run(run_name=run_name)  # type: ignore[attr-defined]

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params) # type: ignore[attr-defined]

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step) # type: ignore[attr-defined]

    def log_artifact(self, path: str) -> None:
        mlflow.log_artifact(path) # type: ignore[attr-defined]

    def end_run(self) -> None:
        if self._run:
            mlflow.end_run()  # type: ignore[attr-defined]
            self._run = None

    @property
    def run_id(self) -> str | None:
        return self._run.info.run_id if self._run else None
