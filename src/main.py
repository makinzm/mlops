"""
MLOps CLI エントリーポイント。

Hydra で設定を読み込み、usecase に応じてインフラを DI して UseCase を実行する。
Kaggle 認証は ~/.kaggle/access_token に保存したトークンを使用する。

実行例:
    uv run python -m src usecase=download_dataset downloader=kaggle
    uv run python -m src usecase=automatically_eda competition=titanic
    uv run python -m src usecase=preprocess recipe=base
    uv run python -m src usecase=train recipe=lgbm
    uv run python -m src usecase=inference recipe=titanic_ensemble
    uv run python -m src usecase=pipeline recipe=all_after_download
"""

import logging
import os
from pathlib import Path
from typing import cast

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.presentation.registry import dispatch

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
)

_CONF_DIR = str(Path(__file__).parent.parent / "conf")


@hydra.main(config_path=_CONF_DIR, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    from src.infrastructure.logger.python_logger import PythonAppLogger

    logger = PythonAppLogger(__name__)
    usecase_name: str = cfg.get("usecase", "download_dataset")

    # presentation 層で KAGGLE_USERNAME を解決して cfg に注入する
    # usecase 層は os に依存できないため、ここで一括処理する（caution.md: struct mode 回避）
    if not cfg.get("platform_username"):
        cfg = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))
        cfg.platform_username = os.environ.get("KAGGLE_USERNAME", "")

    dispatch(usecase_name, cfg, logger)


if __name__ == "__main__":
    main()
