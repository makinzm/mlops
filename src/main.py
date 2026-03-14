"""
MLOps CLI エントリーポイント。

Hydra で設定を読み込み、usecase に応じてインフラを DI して UseCase を実行する。
Kaggle 認証は ~/.kaggle/access_token に保存したトークンを使用する。

実行例:
    uv run python -m src usecase=download_dataset downloader=kaggle
    uv run python -m src usecase=automatically_eda competition=titanic
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


def _resolve_analyzer(cfg: DictConfig) -> object:
    from src.infrastructure.analyzer.pandas_analyzer import PandasAnalyzer
    from src.infrastructure.repository.git import GitRepositoryImpl

    return PandasAnalyzer(cfg, GitRepositoryImpl())


@hydra.main(config_path=_CONF_DIR, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    from src.infrastructure.logger.python_logger import PythonAppLogger

    logger = PythonAppLogger(__name__)
    usecase_name: str = cfg.get("usecase", "download_dataset")

    if usecase_name == "download_dataset":
        from src.usecase.data_acquisition.download_dataset import DownloadDatasetUseCase

        try:
            downloader = _resolve_downloader(cfg)
        except Exception:
            logger.error("ダウンローダーの初期化に失敗しました", exc_info=True)
            raise
        DownloadDatasetUseCase(downloader, logger).execute()  # type: ignore[arg-type]

    elif usecase_name == "automatically_eda":
        from src.usecase.eda.automatically_eda import AutomaticallyEDAUseCase

        AutomaticallyEDAUseCase(_resolve_analyzer(cfg), logger).execute()  # type: ignore[arg-type]

    else:
        raise ValueError(
            f"Unknown usecase: {usecase_name!r}. Supported: 'download_dataset', 'automatically_eda'"
        )


if __name__ == "__main__":
    main()
