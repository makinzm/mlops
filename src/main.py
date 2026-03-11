"""
MLOps CLI エントリーポイント。

Hydra で設定を読み込み、downloader.type に応じてインフラを DI して UseCase を実行する。
Kaggle 認証は ~/.kaggle/access_token に保存したトークンを使用する。

実行例:
    uv run python -m src usecase=download_dataset downloader=kaggle
    uv run python -m src downloader.dataset=owner/name
"""

import logging
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
)

_CONF_DIR = str(Path(__file__).parent.parent / "conf")


def _resolve_downloader(cfg: DictConfig) -> object:
    from src.infrastructure.repository.git import GitRepositoryImpl

    git_repo = GitRepositoryImpl()
    downloader_type = cfg.downloader.type
    if downloader_type == "kaggle":
        from src.infrastructure.downloader.kaggle import KaggleDownloader

        return KaggleDownloader(cfg, git_repo)
    else:
        raise ValueError(f"Unknown downloader type: {downloader_type!r}. Supported: 'kaggle'")


@hydra.main(config_path=_CONF_DIR, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    from src.infrastructure.logger.python_logger import PythonAppLogger
    from src.usecase.data_acquisition.download_dataset import DownloadDatasetUseCase

    logger = PythonAppLogger(__name__)
    try:
        downloader = _resolve_downloader(cfg)
        DownloadDatasetUseCase(downloader, logger).execute()  # type: ignore[arg-type]
    except Exception:
        raise


if __name__ == "__main__":
    main()
