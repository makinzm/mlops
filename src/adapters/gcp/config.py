from dataclasses import dataclass


@dataclass
class GCPConfig:
    """GCP infrastructure configuration"""

    project_id: str
    region: str
    bucket: str
    artifact_registry: str | None = None
    mlflow_tracking_uri: str | None = None

    @property
    def gcs_base_uri(self) -> str:
        return f"gs://{self.bucket}"

    @property
    def data_uri(self) -> str:
        return f"{self.gcs_base_uri}/data"

    @property
    def models_uri(self) -> str:
        return f"{self.gcs_base_uri}/models"
