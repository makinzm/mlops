"""
MLOps CLI エントリーポイント。

Hydra で設定を読み込み、data_from に応じてインフラを DI して UseCase を実行する。
python-dotenv で .env を自動ロードするため、KAGGLE_API_TOKEN 等の手動 export は不要。

実行例:
    uv run python -m src usecase=download_dataset
    uv run python -m src usecase=download_dataset kaggle.dataset=owner/name
"""

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

load_dotenv()


def _resolve_downloader(cfg: DictConfig) -> object:
    data_from = cfg.data_from
    if data_from == "kaggle":
        from src.infrastructure.downloader.kaggle import KaggleDownloader

        return KaggleDownloader(cfg)
    else:
        raise ValueError(f"Unknown data_from: {data_from!r}. Supported: 'kaggle'")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    from src.usecase.data_acquisition.download_dataset import DownloadDatasetUseCase

    downloader = _resolve_downloader(cfg)
    result = DownloadDatasetUseCase(downloader).execute()  # type: ignore[arg-type]
    print(f"Downloaded to: {result.output_dir}")
    print(f"Files: {len(result.files)}")
    print(f"Commit: {result.commit_hash}")


if __name__ == "__main__":
    main()
