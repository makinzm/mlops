from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class KaggleConfig:
    """Kaggle infrastructure configuration"""

    competition: str
    username: str
    data_path: Path = field(default_factory=lambda: Path("/kaggle/input"))
    output_path: Path = field(default_factory=lambda: Path("/kaggle/working"))
    dataset_id: str | None = None
    model_id: str | None = None

    @property
    def competition_data_path(self) -> Path:
        return self.data_path / self.competition

    @property
    def dataset_slug(self) -> str:
        if self.dataset_id:
            return f"{self.username}/{self.dataset_id}"
        return f"{self.username}/{self.competition}-code"

    @property
    def model_slug(self) -> str:
        if self.model_id:
            return f"{self.username}/{self.model_id}"
        return f"{self.username}/{self.competition}-model"
