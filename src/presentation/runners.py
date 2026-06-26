"""UseCase ランナー関数群。

main.py から分離した _run_* 関数を提供する。
各関数は Hydra DictConfig を受け取り、DI を行って UseCase を実行する。
PipelineUseCase にもこれらの関数が注入される。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.presentation.cloud_config import (
    ensure_cloud_config,
    load_trainer_cfgs_safe,
    resolve_manifest_path,
)
from src.presentation.kaggle_auth import authenticate_kaggle_api

_CONF_DIR = str(Path(__file__).parent.parent.parent / "conf")


def _resolve_downloader(cfg: DictConfig) -> object:
    from src.infrastructure.repository.git import GitRepositoryImpl

    git_repo = GitRepositoryImpl()
    source = str(cfg.get("source", "kaggle"))
    if source == "kaggle":
        from src.infrastructure.downloader.kaggle import KaggleDownloader

        return KaggleDownloader(cfg, git_repo)
    else:
        raise ValueError(f"Unknown source: {source!r}. Supported: 'kaggle'")


def _parse_analyses(steps_cfg: Any) -> list[Any]:
    """OmegaConf の steps リストを AnalysisStep リストに変換する。"""
    from src.domain.data.eda import AnalysisStep

    raw = OmegaConf.to_container(steps_cfg, resolve=True)
    steps = []
    for item in raw:  # ty:ignore[not-iterable]
        d = dict(item)  # ty:ignore[no-matching-overload]
        step_type = d.pop("type")
        steps.append(AnalysisStep(type=step_type, params=d))
    return steps


def _resolve_analyzers(cfg: DictConfig) -> list[object]:
    """cfg.analyze の各エントリに対応するアナライザーを生成して返す。"""
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


def run_download(cfg: DictConfig, logger: Any = None) -> None:
    """ダウンロード UseCase を実行する。"""
    from src.usecase.data_acquisition.download_dataset import DownloadDatasetUseCase

    if logger is None:
        logger = logging.getLogger(__name__)
    try:
        downloader = _resolve_downloader(cfg)
    except Exception:
        logger.error("ダウンローダーの初期化に失敗しました", exc_info=True)
        raise
    DownloadDatasetUseCase(downloader, logger).execute()  # ty:ignore[invalid-argument-type]


def run_automatically_eda(cfg: DictConfig, logger: Any = None) -> None:
    """EDA UseCase を実行する。"""
    from src.usecase.eda.automatically_eda import AutomaticallyEDAUseCase

    if logger is None:
        logger = logging.getLogger(__name__)
    analyzers = _resolve_analyzers(cfg)
    AutomaticallyEDAUseCase(analyzers, logger).execute()  # ty:ignore[invalid-argument-type]


def run_preprocess(cfg: DictConfig, logger: Any = None) -> None:
    """前処理 UseCase を実行する。"""
    from src.infrastructure.executor.factory import ExecutorFactory
    from src.infrastructure.preprocessor.cv_splitter import CVSplitter
    from src.infrastructure.preprocessor.input_loader import InputLoader
    from src.infrastructure.preprocessor.visualizer import PipelineVisualizer
    from src.infrastructure.repository.git import GitRepositoryImpl
    from src.usecase.preprocessing.pipeline_loader import load_pipeline_cfgs
    from src.usecase.preprocessing.preprocess import PreprocessUseCase

    if logger is None:
        logger = logging.getLogger(__name__)
    git_repo = GitRepositoryImpl()
    pipeline_cfgs = load_pipeline_cfgs(cfg, Path(_CONF_DIR))
    for pipeline_cfg in pipeline_cfgs:
        executor_type = str(pipeline_cfg.get("executor", {}).get("type", "local"))
        executor, is_fallback = ExecutorFactory.build_with_fallback(executor_type)
        result = PreprocessUseCase(
            pipeline_cfg,
            executor=executor,
            git_repo=git_repo,
            input_loader=InputLoader(),
            cv_splitter=CVSplitter(),
            visualizer=PipelineVisualizer(),
            executor_fallback=is_fallback,
            executor_requested=executor_type if is_fallback else None,
        ).execute()
        logger.info(
            f"前処理完了[{pipeline_cfg.get('job_id', '?')}]: "
            f"output_path={result.output_path}, steps={len(result.step_results)}"
        )


def run_train(cfg: DictConfig, logger: Any = None) -> None:
    """学習 UseCase を実行する。"""
    from src.infrastructure.repository.git import GitRepositoryImpl
    from src.infrastructure.trainer.lgbm_trainer import LightGBMTrainer
    from src.usecase.training.train import TrainUseCase
    from src.usecase.training.trainer_loader import load_trainer_cfgs

    if logger is None:
        logger = logging.getLogger(__name__)
    git_repo = GitRepositoryImpl()
    trainer_cfgs = load_trainer_cfgs(cfg, Path(_CONF_DIR))
    for trainer_cfg in trainer_cfgs:
        trainer_type: str = trainer_cfg.trainer.type
        if trainer_type == "lgbm":
            trainer = LightGBMTrainer(trainer_cfg)
        elif trainer_type == "vision":
            from src.infrastructure.trainer.vision_trainer import VisionTrainer

            raw_cfg: dict[str, Any] = OmegaConf.to_container(trainer_cfg, resolve=True)  # ty:ignore[invalid-assignment]
            trainer = VisionTrainer(raw_cfg)
        elif trainer_type == "audio":
            from src.infrastructure.trainer.audio_trainer import AudioTrainer

            trainer = AudioTrainer()
        else:
            raise ValueError(
                f"trainer.type='{trainer_type}' は未登録です。"
                " 登録済み: ['lgbm', 'vision', 'audio']"
            )
        train_result = TrainUseCase(trainer_cfg, trainer=trainer, git_repo=git_repo).execute()
        logger.info(
            f"学習完了[{train_result.job_id}]: "
            f"CV {train_result.metric}="
            f"{train_result.cv_mean_score:.4f} ± {train_result.cv_std_score:.4f}"
        )


def run_job_train(cfg: DictConfig, logger: Any = None) -> None:
    """学習ジョブ同期実行 UseCase を実行する。"""
    from src.infrastructure.gcp.storage import GCSRepositoryImpl
    from src.infrastructure.gcp.vertex_ai import VertexAIRepositoryImpl
    from src.infrastructure.repository.git import GitRepositoryImpl
    from src.usecase.training.job_train import JobTrainUseCase
    from src.usecase.training.trainer_loader import load_trainer_cfgs

    if logger is None:
        logger = logging.getLogger(__name__)
    cfg = ensure_cloud_config(cfg, _CONF_DIR)
    git_repo = GitRepositoryImpl()
    trainer_cfgs = load_trainer_cfgs(cfg, Path(_CONF_DIR))
    trainer_cfg = trainer_cfgs[0]
    if trainer_cfg.get("infra") is None:
        trainer_cfg = DictConfig(OmegaConf.merge(trainer_cfg, {"infra": cfg.infra}))
    gcs = GCSRepositoryImpl(project=str(trainer_cfg.infra.project))
    vertex = VertexAIRepositoryImpl(
        project=str(trainer_cfg.infra.project),
        region=str(trainer_cfg.infra.region),
        staging_bucket=str(trainer_cfg.infra.staging_bucket),
    )
    result = JobTrainUseCase(
        cfg=trainer_cfg,
        object_storage=gcs,
        training_job=vertex,
        git_repo=git_repo,
    ).execute()
    logger.info(
        f"学習ジョブ完了[{result.job_id}]: "
        f"job={result.job_name}, "
        f"local_model_dir={result.local_model_dir}"
    )


def run_job_submit(cfg: DictConfig, logger: Any = None) -> None:
    """学習ジョブ非同期送信 UseCase を実行する。"""
    from src.infrastructure.gcp.storage import GCSRepositoryImpl
    from src.infrastructure.gcp.vertex_ai import VertexAIRepositoryImpl
    from src.infrastructure.repository.git import GitRepositoryImpl
    from src.usecase.training.job_submit import JobSubmitUseCase
    from src.usecase.training.trainer_loader import load_trainer_cfgs

    if logger is None:
        logger = logging.getLogger(__name__)
    cfg = ensure_cloud_config(cfg, _CONF_DIR)
    git_repo = GitRepositoryImpl()
    trainer_cfgs = load_trainer_cfgs(cfg, Path(_CONF_DIR))
    trainer_cfg = trainer_cfgs[0]
    if trainer_cfg.get("infra") is None:
        trainer_cfg = DictConfig(OmegaConf.merge(trainer_cfg, {"infra": cfg.infra}))
    gcs = GCSRepositoryImpl(project=str(trainer_cfg.infra.project))
    vertex = VertexAIRepositoryImpl(
        project=str(trainer_cfg.infra.project),
        region=str(trainer_cfg.infra.region),
        staging_bucket=str(trainer_cfg.infra.staging_bucket),
    )
    result = JobSubmitUseCase(
        cfg=trainer_cfg,
        object_storage=gcs,
        training_job=vertex,
        git_repo=git_repo,
    ).execute()
    logger.info(
        f"学習ジョブ送信完了[{result.job_id}]: "
        f"job={result.job_name}, manifest={result.manifest_path}"
    )


def run_job_download(cfg: DictConfig, logger: Any = None) -> None:
    """学習ジョブ結果ダウンロード UseCase を実行する。"""
    from src.infrastructure.gcp.storage import GCSRepositoryImpl
    from src.infrastructure.gcp.vertex_ai import VertexAIRepositoryImpl
    from src.usecase.training.job_download import JobDownloadUseCase

    if logger is None:
        logger = logging.getLogger(__name__)
    cfg = ensure_cloud_config(cfg, _CONF_DIR)
    gcs = GCSRepositoryImpl(project=str(cfg.infra.project))
    vertex = VertexAIRepositoryImpl(
        project=str(cfg.infra.project),
        region=str(cfg.infra.region),
        staging_bucket=str(cfg.infra.staging_bucket),
    )
    manifest_path = resolve_manifest_path(cfg, _CONF_DIR)
    logger.info(f"Using manifest: {manifest_path}")
    output_dir_raw = cfg.get("output_dir")
    if output_dir_raw is None or str(output_dir_raw) == "None":
        trainer_cfgs = load_trainer_cfgs_safe(cfg, _CONF_DIR)
        output_dir = Path(str(trainer_cfgs[0].output_dir))
    else:
        output_dir = Path(str(output_dir_raw))
    result = JobDownloadUseCase(
        manifest_path=manifest_path,
        object_storage=gcs,
        training_job=vertex,
        output_dir=output_dir,
    ).execute()
    logger.info(
        f"モデルダウンロード完了[{result.job_id}]: local_model_dir={result.local_model_dir}"
    )


def run_inference(cfg: DictConfig, logger: Any = None) -> None:
    """推論 UseCase を実行する。"""
    from src.infrastructure.inference.lgbm_inferencer import LightGBMInferencer
    from src.infrastructure.repository.git import GitRepositoryImpl
    from src.usecase.inference.inference import InferenceUseCase
    from src.usecase.inference.inference_loader import load_inference_cfgs

    if logger is None:
        logger = logging.getLogger(__name__)
    git_repo = GitRepositoryImpl()
    inference_cfgs = load_inference_cfgs(cfg, Path(_CONF_DIR))
    for inference_cfg in inference_cfgs:
        inferencer_type: str = str(inference_cfg.get("inferencer_type", "lgbm"))
        if inferencer_type == "vision":
            from src.infrastructure.inference.vision_inferencer import VisionInferencer

            inferencer = VisionInferencer()
        else:
            inferencer = LightGBMInferencer()
        submission_path = InferenceUseCase(inferencer=inferencer, git_repo=git_repo).run(
            inference_cfg
        )
        logger.info(f"推論完了[{inference_cfg.get('job_id', '?')}]: {submission_path}")


def run_pipeline(cfg: DictConfig, logger: Any = None) -> None:
    """パイプライン UseCase を実行する。"""
    from src.usecase.pipeline.pipeline import PipelineUseCase
    from src.usecase.pipeline.pipeline_loader import load_pipeline_recipe_cfg

    if logger is None:
        logger = logging.getLogger(__name__)
    pipeline_cfg = load_pipeline_recipe_cfg(cfg, Path(_CONF_DIR))

    def _pipeline_runner(runner_func):
        """Pipeline 用ラッパー: (cfg) → (cfg, logger=None) の変換。"""

        def wrapper(step_cfg: DictConfig) -> None:
            runner_func(step_cfg)

        return wrapper

    PipelineUseCase(
        run_preprocess=_pipeline_runner(run_preprocess),
        run_train=_pipeline_runner(run_train),
        run_inference=_pipeline_runner(run_inference),
        conf_dir=Path(_CONF_DIR),
        run_job_train=_pipeline_runner(run_job_train),
        run_job_submit=_pipeline_runner(run_job_submit),
        run_job_download=_pipeline_runner(run_job_download),
        run_update_source_dataset=_pipeline_runner(run_update_source_dataset),
        run_push_notebook=_pipeline_runner(run_push_notebook),
        run_download_dataset=_pipeline_runner(run_download),
    ).run(pipeline_cfg)
    logger.info(f"パイプライン完了[{pipeline_cfg.get('job_id', '?')}]")


def run_push_notebook(cfg: DictConfig, logger: Any = None) -> None:
    """Notebook push UseCase を実行する。

    conf/usecase/push_notebook.yaml は routing-only のため、
    cfg.recipe が指定されていれば conf/competition/{name}/notebook/{recipe}.yaml を
    マージしてから実行する（pipeline 経由で notebook が既に dict 指定済みの場合は素通し）。
    """
    from src.usecase.notebook.notebook_loader import load_notebook_recipe_cfg
    from src.usecase.notebook.push_notebook import PushNotebookUseCase

    if logger is None:
        logger = logging.getLogger(__name__)
    merged = load_notebook_recipe_cfg(cfg, Path(_CONF_DIR))
    kaggle_api = authenticate_kaggle_api()
    result = PushNotebookUseCase(cfg=merged, platform_api=kaggle_api).execute()
    logger.info(f"Notebook push 完了: notebook={result.notebook_path}")


def run_update_source_dataset(cfg: DictConfig, logger: Any = None) -> None:
    """update_source_dataset UseCase を実行する。"""
    from src.infrastructure.kaggle.source_dataset import KaggleSourceDatasetRepository
    from src.usecase.source_dataset.update_source_dataset import UpdateSourceDatasetUseCase

    if logger is None:
        logger = logging.getLogger(__name__)
    kaggle_api = authenticate_kaggle_api()
    repository = KaggleSourceDatasetRepository(kaggle_api=kaggle_api)
    UpdateSourceDatasetUseCase(cfg=cfg, repository=repository).execute()
    logger.info("update_source_dataset 完了")


def run_create_source_dataset(cfg: DictConfig, logger: Any = None) -> None:
    """create_source_dataset UseCase を実行する。"""
    from src.infrastructure.kaggle.source_dataset import KaggleSourceDatasetRepository
    from src.usecase.source_dataset.create_source_dataset import CreateSourceDatasetUseCase

    if logger is None:
        logger = logging.getLogger(__name__)
    kaggle_api = authenticate_kaggle_api()
    repository = KaggleSourceDatasetRepository(kaggle_api=kaggle_api)
    CreateSourceDatasetUseCase(cfg=cfg, repository=repository).execute()
    logger.info("create_source_dataset 完了")


def run_gradcam(cfg: DictConfig, logger: Any = None) -> None:
    """GradCAM UseCase を実行する。"""
    from src.infrastructure.analyzer.gradcam_analyzer import GradCAMAnalyzerImpl
    from src.infrastructure.repository.git import GitRepositoryImpl
    from src.usecase.analysis.gradcam import GradCAMUseCase

    if logger is None:
        logger = logging.getLogger(__name__)
    git_repo = GitRepositoryImpl()
    raw_cfg: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # ty:ignore[invalid-assignment]
    results = GradCAMUseCase(
        cfg=raw_cfg,
        analyzer=GradCAMAnalyzerImpl(),
        git_repo=git_repo,
    ).execute()
    logger.info(f"GradCAM 完了: {len(results)} 画像分析")
