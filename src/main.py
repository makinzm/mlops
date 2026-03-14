"""
MLOps CLI エントリーポイント。

Hydra で設定を読み込み、usecase に応じてインフラを DI して UseCase を実行する。
Kaggle 認証は ~/.kaggle/access_token に保存したトークンを使用する。

実行例:
    uv run python -m src usecase=download_dataset downloader=kaggle
    uv run python -m src usecase=automatically_eda competition=titanic
    uv run python -m src usecase=automatically_eda competition=titanic \\
        "analyze.pandas.steps=[{type:basic_stats},{type:group_stats,group_by:Survived}]"
"""

import logging
from pathlib import Path
from typing import Any

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

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


def _parse_analyses(steps_cfg: Any) -> list[Any]:
    """OmegaConf の steps リストを AnalysisStep リストに変換する。"""
    from src.domain.data.eda import AnalysisStep

    raw = OmegaConf.to_container(steps_cfg, resolve=True)
    steps = []
    for item in raw:  # type: ignore[union-attr]
        d = dict(item)  # type: ignore[arg-type]
        step_type = d.pop("type")
        steps.append(AnalysisStep(type=step_type, params=d))
    return steps


def _resolve_analyzers(cfg: DictConfig) -> list[object]:
    """cfg.analyze の各エントリに対応するアナライザーを生成して返す。

    形式:
        analyze:
          pandas:
            output_format: parquet  # or csv
            steps:
              - type: basic_stats
          polars:
            steps:
              - type: distributions
    """
    from src.infrastructure.repository.git import GitRepositoryImpl

    commit_hash = GitRepositoryImpl().get_commit_hash()
    analyzers: list[object] = []

    for analyzer_type, analyzer_cfg in cfg.analyze.items():
        analyses = _parse_analyses(analyzer_cfg.steps)

        if analyzer_type == "pandas":
            from src.infrastructure.analyzer.pandas_analyzer import PandasAnalyzer

            output_format: str = getattr(analyzer_cfg, "output_format", "parquet")
            analyzers.append(PandasAnalyzer(cfg, commit_hash, analyses, output_format))

        elif analyzer_type == "polars":
            from src.infrastructure.analyzer.polars_analyzer import PolarsAnalyzer

            analyzers.append(PolarsAnalyzer(cfg, commit_hash, analyses))

        else:
            raise ValueError(
                f"Unknown analyzer type: {analyzer_type!r}. Supported: 'pandas', 'polars'"
            )

    return analyzers


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

        analyzers = _resolve_analyzers(cfg)
        AutomaticallyEDAUseCase(analyzers, logger).execute()  # type: ignore[arg-type]

    elif usecase_name == "preprocess":
        from src.usecase.preprocessing.preprocess import PreprocessUseCase

        result = PreprocessUseCase(cfg).execute()
        logger.info(
            f"前処理完了: output_path={result.output_path}, "
            f"steps={len(result.step_results)}, fallback={result.executor_fallback}"
        )

    else:
        raise ValueError(
            f"Unknown usecase: {usecase_name!r}. "
            "Supported: 'download_dataset', 'automatically_eda', 'preprocess'"
        )


if __name__ == "__main__":
    main()
