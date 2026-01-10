from io import BytesIO
from pathlib import Path

import pandas as pd
from google.cloud import storage

from src.adapters.gcp.config import GCPConfig


class GCSDataStore:
    """Google Cloud Storage implementation of DataStore port"""

    def __init__(self, config: GCPConfig) -> None:
        self.config = config
        self.client = storage.Client(project=config.project_id)
        self.bucket = self.client.bucket(config.bucket)

    def _get_blob(self, prefix: str, name: str) -> storage.Blob:
        return self.bucket.blob(f"data/{prefix}/{name}.csv")

    def load_raw(self, name: str) -> pd.DataFrame:
        blob = self._get_blob("raw", name)
        content = blob.download_as_bytes()
        return pd.read_csv(BytesIO(content))

    def save_processed(self, df: pd.DataFrame, name: str) -> Path:
        blob = self._get_blob("processed", name)
        blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")
        return Path(f"gs://{self.config.bucket}/data/processed/{name}.csv")

    def load_processed(self, name: str) -> pd.DataFrame:
        blob = self._get_blob("processed", name)
        content = blob.download_as_bytes()
        return pd.read_csv(BytesIO(content))

    def save_features(self, df: pd.DataFrame, name: str) -> Path:
        blob = self._get_blob("features", name)
        blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")
        return Path(f"gs://{self.config.bucket}/data/features/{name}.csv")

    def load_features(self, name: str) -> pd.DataFrame:
        blob = self._get_blob("features", name)
        content = blob.download_as_bytes()
        return pd.read_csv(BytesIO(content))
