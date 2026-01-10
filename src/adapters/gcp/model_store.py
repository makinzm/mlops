import json
import pickle
from io import BytesIO
from pathlib import Path
from typing import Any

from google.cloud import storage

from src.adapters.gcp.config import GCPConfig


class GCSModelStore:
    """GCS implementation of ModelStore port"""

    def __init__(self, config: GCPConfig) -> None:
        self.config = config
        self.client = storage.Client(project=config.project_id)
        self.bucket = self.client.bucket(config.bucket)

    def save(
        self,
        model: Any,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        # Save model
        blob = self.bucket.blob(f"models/{name}/model.pkl")
        buffer = BytesIO()
        pickle.dump({"model": model, "metadata": metadata}, buffer)
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type="application/octet-stream")

        # Save metadata as JSON
        if metadata:
            meta_blob = self.bucket.blob(f"models/{name}/metadata.json")
            meta_blob.upload_from_string(
                json.dumps(metadata, indent=2),
                content_type="application/json",
            )

        return Path(f"gs://{self.config.bucket}/models/{name}/model.pkl")

    def load(self, name: str) -> Any:
        blob = self.bucket.blob(f"models/{name}/model.pkl")
        content = blob.download_as_bytes()
        data = pickle.loads(content)
        return data["model"]

    def list_models(self) -> list[str]:
        blobs = self.client.list_blobs(self.bucket, prefix="models/", delimiter="/")
        # Extract model names from prefixes
        models = []
        for page in blobs.pages:
            for prefix in page.prefixes:
                models.append(prefix.rstrip("/").split("/")[-1])
        return models

    def get_model_uri(self, name: str) -> str:
        """Get GCS URI for model"""
        return f"gs://{self.config.bucket}/models/{name}/model.pkl"
