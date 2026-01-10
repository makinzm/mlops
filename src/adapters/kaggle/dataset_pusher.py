import json
import shutil
import subprocess
from pathlib import Path

from src.adapters.kaggle.config import KaggleConfig


class KaggleDatasetPusher:
    """Push code and data to Kaggle Datasets

    This enables using GitHub code in Kaggle Notebooks by:
    1. Packaging src/, configs/ as a Kaggle Dataset
    2. Versioning with each commit
    """

    def __init__(self, config: KaggleConfig) -> None:
        self.config = config

    def package_and_push(
        self,
        source_dirs: list[Path],
        version_notes: str = "",
        title: str | None = None,
    ) -> str:
        """Package source directories and push to Kaggle Datasets"""
        # Create temp directory for dataset
        package_dir = Path("/tmp/kaggle-dataset-package")
        if package_dir.exists():
            shutil.rmtree(package_dir)
        package_dir.mkdir(parents=True)

        # Copy source files
        for src_dir in source_dirs:
            if src_dir.is_dir():
                dest = package_dir / src_dir.name
                shutil.copytree(src_dir, dest)
            else:
                shutil.copy2(src_dir, package_dir)

        # Create dataset-metadata.json
        metadata = {
            "title": title or f"{self.config.competition} Code",
            "id": self.config.dataset_slug,
            "licenses": [{"name": "CC0-1.0"}],
        }

        meta_path = package_dir / "dataset-metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Check if dataset exists
        check_result = subprocess.run(
            ["kaggle", "datasets", "status", self.config.dataset_slug],
            capture_output=True,
            text=True,
        )

        if "404" in check_result.stderr or "Not found" in check_result.stderr:
            # Create new dataset
            cmd = ["kaggle", "datasets", "create", "-p", str(package_dir)]
        else:
            # Update existing dataset
            cmd = [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(package_dir),
                "-m",
                version_notes or "Update",
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Kaggle dataset push failed: {result.stderr}")

        return self.config.dataset_slug
