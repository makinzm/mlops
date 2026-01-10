import subprocess
from pathlib import Path

import pandas as pd

from src.adapters.kaggle.config import KaggleConfig


class KaggleServingGateway:
    """Kaggle competition submission implementation of ServingGateway port"""

    def __init__(self, config: KaggleConfig) -> None:
        self.config = config

    def submit(self, predictions: pd.DataFrame, name: str) -> str:
        """Submit predictions to Kaggle competition"""
        # Save predictions to file
        submission_path = self.config.output_path / "submissions" / f"{name}.csv"
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(submission_path, index=False)

        # Submit via kaggle CLI
        result = subprocess.run(
            [
                "kaggle",
                "competitions",
                "submit",
                "-c",
                self.config.competition,
                "-f",
                str(submission_path),
                "-m",
                name,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Kaggle submission failed: {result.stderr}")

        return f"{self.config.competition}:{name}"

    def get_submission_status(self, submission_id: str) -> dict:
        """Get submission status from Kaggle"""
        result = subprocess.run(
            [
                "kaggle",
                "competitions",
                "submissions",
                "-c",
                self.config.competition,
                "--csv",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to get submissions: {result.stderr}")

        lines = result.stdout.strip().split("\n")
        if len(lines) > 1:
            return {"status": "submitted", "raw": lines[1]}

        return {"status": "unknown"}

    def export_for_serving(self, model_path: Path) -> Path:
        """Export model for Kaggle serving (not applicable for competitions)"""
        return model_path
