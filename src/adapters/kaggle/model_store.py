import json
import pickle
import subprocess
from pathlib import Path
from typing import Any

from src.adapters.kaggle.config import KaggleConfig


class KaggleModelStore:
    """Kaggle Models API implementation of ModelStore port

    Supports both:
    - Local model storage (within Kaggle notebook)
    - Kaggle Models registry (public model sharing)
    """

    def __init__(self, config: KaggleConfig) -> None:
        self.config = config
        self.local_path = config.output_path / "models"
        self.local_path.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: Any,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        model_dir = self.local_path / name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "metadata": metadata}, f)

        # Save metadata
        if metadata:
            meta_path = model_dir / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

        return model_path

    def load(self, name: str) -> Any:
        model_path = self.local_path / name / "model.pkl"
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        return data["model"]

    def list_models(self) -> list[str]:
        return [p.name for p in self.local_path.iterdir() if p.is_dir()]

    def push_to_kaggle_models(
        self,
        name: str,
        instance_name: str,
        version_notes: str = "",
        framework: str = "other",
    ) -> str:
        """Push model to Kaggle Models registry"""
        model_dir = self.local_path / name

        # Create model-metadata.json for Kaggle Models
        metadata = {
            "ownerSlug": self.config.username,
            "modelSlug": self.config.model_slug.split("/")[-1],
            "instanceSlug": instance_name,
            "framework": framework,
            "overview": version_notes,
            "licenseName": "Apache 2.0",
        }

        meta_path = model_dir / "model-metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Push using kaggle CLI
        result = subprocess.run(
            [
                "kaggle",
                "models",
                "instances",
                "version",
                "-p",
                str(model_dir),
                "-n",
                version_notes or "New version",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Kaggle model push failed: {result.stderr}")

        return f"{self.config.model_slug}/{instance_name}"
