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
from typing import Any, cast

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


def _run_download(cfg: DictConfig) -> None:
    """ダウンロード UseCase を実行する（Pipeline から呼ばれる）。"""
    from src.usecase.data_acquisition.download_dataset import DownloadDatasetUseCase

    logger = logging.getLogger(__name__)
    try:
        downloader = _resolve_downloader(cfg)
    except Exception:
        logger.error("ダウンローダーの初期化に失敗しました", exc_info=True)
        raise
    DownloadDatasetUseCase(downloader, logger).execute()  # ty:ignore[invalid-argument-type]


def _run_preprocess(cfg: DictConfig) -> None:
    """前処理 UseCase を実行する（Pipeline から呼ばれる）。"""
    from src.infrastructure.executor.factory import ExecutorFactory
    from src.infrastructure.preprocessor.cv_splitter import CVSplitter
    from src.infrastructure.preprocessor.input_loader import InputLoader
    from src.infrastructure.preprocessor.visualizer import PipelineVisualizer
    from src.infrastructure.repository.git import GitRepositoryImpl
    from src.usecase.preprocessing.pipeline_loader import load_pipeline_cfgs
    from src.usecase.preprocessing.preprocess import PreprocessUseCase

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


def _run_train(cfg: DictConfig) -> None:
    """学習 UseCase を実行する（Pipeline から呼ばれる）。"""
    from src.infrastructure.repository.git import GitRepositoryImpl
    from src.infrastructure.trainer.lgbm_trainer import LightGBMTrainer
    from src.usecase.training.train import TrainUseCase
    from src.usecase.training.trainer_loader import load_trainer_cfgs

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
        else:
            raise ValueError(
                f"trainer.type='{trainer_type}' は未登録です。 登録済み: ['lgbm', 'vision']"
            )
        train_result = TrainUseCase(trainer_cfg, trainer=trainer, git_repo=git_repo).execute()
        logger.info(
            f"学習完了[{train_result.job_id}]: "
            f"CV {train_result.metric}="
            f"{train_result.cv_mean_score:.4f} ± {train_result.cv_std_score:.4f}"
        )


def _run_remote_train(cfg: DictConfig) -> None:
    """リモート学習 UseCase を実行する（Pipeline から呼ばれる）。"""
    from src.infrastructure.gcp.storage import GCSRepositoryImpl
    from src.infrastructure.gcp.vertex_ai import VertexAIRepositoryImpl
    from src.infrastructure.repository.git import GitRepositoryImpl
    from src.usecase.training.remote_train import RemoteTrainUseCase
    from src.usecase.training.trainer_loader import load_trainer_cfgs

    logger = logging.getLogger(__name__)
    cfg = _ensure_cloud_config(cfg)
    git_repo = GitRepositoryImpl()
    trainer_cfgs = load_trainer_cfgs(cfg, Path(_CONF_DIR))
    # remote_train は一度に 1 レシピを実行する（最初の設定を使用）
    trainer_cfg = trainer_cfgs[0]
    if trainer_cfg.get("cloud") is None:
        trainer_cfg = DictConfig(OmegaConf.merge(trainer_cfg, {"cloud": cfg.cloud}))
    gcs = GCSRepositoryImpl(project=str(trainer_cfg.cloud.project))
    vertex = VertexAIRepositoryImpl(
        project=str(trainer_cfg.cloud.project),
        region=str(trainer_cfg.cloud.region),
        staging_bucket=str(trainer_cfg.cloud.staging_bucket),
    )
    result = RemoteTrainUseCase(
        cfg=trainer_cfg,
        object_storage=gcs,
        training_job=vertex,
        git_repo=git_repo,
    ).execute()
    logger.info(
        f"リモート学習完了[{result.job_id}]: "
        f"job={result.remote_job_name}, "
        f"local_model_dir={result.local_model_dir}"
    )


def _ensure_cloud_config(cfg: DictConfig) -> DictConfig:
    """cloud 設定が未解決の場合、conf/cloud/vertex.yaml を手動マージする。

    Pipeline 経由の場合、Hydra の defaults 処理が走らないため
    cloud: null のままになることがある。その場合は明示的にロードしてマージする。

    同様に notification 設定も未解決なら conf/notification/slack.yaml をマージする。
    """
    needs_merge = False
    extras: list[object] = []

    if cfg.get("cloud") is None:
        cloud_yaml = Path(_CONF_DIR) / "cloud" / "vertex.yaml"
        if not cloud_yaml.exists():
            raise FileNotFoundError(f"Cloud config not found: {cloud_yaml}")
        extras.append(OmegaConf.load(cloud_yaml))
        needs_merge = True

    if cfg.get("notification") is None:
        slack_yaml = Path(_CONF_DIR) / "notification" / "slack.yaml"
        if slack_yaml.exists():
            extras.append(OmegaConf.load(slack_yaml))
            needs_merge = True

    if not needs_merge:
        return cfg

    base = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for extra in extras:
        base = OmegaConf.merge(base, extra)
    return DictConfig(base)


def _run_vertex_submit(cfg: DictConfig) -> None:
    """Vertex AI ジョブ非同期送信 UseCase を実行する（Pipeline から呼ばれる）。"""
    from src.infrastructure.gcp.storage import GCSRepositoryImpl
    from src.infrastructure.gcp.vertex_ai import VertexAIRepositoryImpl
    from src.infrastructure.repository.git import GitRepositoryImpl
    from src.usecase.training.remote_submit import RemoteSubmitUseCase
    from src.usecase.training.trainer_loader import load_trainer_cfgs

    logger = logging.getLogger(__name__)
    cfg = _ensure_cloud_config(cfg)
    git_repo = GitRepositoryImpl()
    trainer_cfgs = load_trainer_cfgs(cfg, Path(_CONF_DIR))
    trainer_cfg = trainer_cfgs[0]
    # trainer yaml に cloud が含まれない場合、元の cfg から補完する
    if trainer_cfg.get("cloud") is None:
        trainer_cfg = DictConfig(OmegaConf.merge(trainer_cfg, {"cloud": cfg.cloud}))
    gcs = GCSRepositoryImpl(project=str(trainer_cfg.cloud.project))
    vertex = VertexAIRepositoryImpl(
        project=str(trainer_cfg.cloud.project),
        region=str(trainer_cfg.cloud.region),
        staging_bucket=str(trainer_cfg.cloud.staging_bucket),
    )
    result = RemoteSubmitUseCase(
        cfg=trainer_cfg,
        object_storage=gcs,
        training_job=vertex,
        git_repo=git_repo,
    ).execute()
    logger.info(
        f"Vertex AI ジョブ送信完了[{result.job_id}]: "
        f"job={result.remote_job_name}, manifest={result.manifest_path}"
    )


def _resolve_manifest_path(cfg: DictConfig) -> Path:
    """manifest_path を解決する。

    manifest_path が指定されていればそのまま返す。
    未指定の場合は competition + job_id + latest から自動解決する。
    """
    from src.usecase._utils import resolve_latest_dir

    explicit = cfg.get("manifest_path")
    if explicit is not None and str(explicit) != "None":
        return Path(str(explicit))

    # config から自動解決
    competition = str(cfg.competition.name)
    job_id = str(cfg.job_id)
    history_base = str(cfg.get("remote_jobs_history_dir", "remote_jobs_history"))
    latest_dir = resolve_latest_dir(f"{history_base}/{competition}/{job_id}/latest")
    return Path(latest_dir) / "job_manifest.yaml"


def _run_vertex_download(cfg: DictConfig) -> None:
    """Vertex AI モデルダウンロード UseCase を実行する（Pipeline から呼ばれる）。"""
    from src.infrastructure.gcp.storage import GCSRepositoryImpl
    from src.infrastructure.gcp.vertex_ai import VertexAIRepositoryImpl
    from src.usecase.training.remote_download import RemoteDownloadUseCase

    logger = logging.getLogger(__name__)
    cfg = _ensure_cloud_config(cfg)
    gcs = GCSRepositoryImpl(project=str(cfg.cloud.project))
    vertex = VertexAIRepositoryImpl(
        project=str(cfg.cloud.project),
        region=str(cfg.cloud.region),
        staging_bucket=str(cfg.cloud.staging_bucket),
    )
    manifest_path = _resolve_manifest_path(cfg)
    logger.info(f"Using manifest: {manifest_path}")
    output_dir = Path(str(cfg.output_dir))
    result = RemoteDownloadUseCase(
        manifest_path=manifest_path,
        object_storage=gcs,
        training_job=vertex,
        output_dir=output_dir,
    ).execute()
    logger.info(
        f"モデルダウンロード完了[{result.job_id}]: local_model_dir={result.local_model_dir}"
    )


def _run_update_source_dataset_pipeline(cfg: DictConfig) -> None:
    """update_source_dataset UseCase を Pipeline から実行する。"""
    from src.infrastructure.kaggle.source_dataset import KaggleSourceDatasetRepository
    from src.usecase.source_dataset.update_source_dataset import UpdateSourceDatasetUseCase

    try:
        from kaggle.api.kaggle_api_extended import (
            KaggleApi as KaggleApiExtended,
        )
    except SystemExit as e:
        raise RuntimeError(
            "Kaggle 認証に失敗しました。~/.kaggle/access_token を確認してください。"
        ) from e

    kaggle_api = KaggleApiExtended()
    try:
        kaggle_api.authenticate()
    except SystemExit as e:
        raise RuntimeError(
            "Kaggle 認証に失敗しました。~/.kaggle/access_token を確認してください。"
        ) from e

    repository = KaggleSourceDatasetRepository(kaggle_api=kaggle_api)
    UpdateSourceDatasetUseCase(cfg=cfg, repository=repository).execute()
    logging.getLogger(__name__).info("update_source_dataset 完了")


def _run_push_notebook_pipeline(cfg: DictConfig) -> None:
    """push_notebook UseCase を Pipeline から実行する。"""
    from src.usecase.notebook.push_notebook import PushNotebookUseCase

    try:
        from kaggle.api.kaggle_api_extended import (
            KaggleApi as KaggleApiExtended,
        )
    except SystemExit as e:
        raise RuntimeError(
            "Kaggle 認証に失敗しました。~/.kaggle/access_token を確認してください。"
        ) from e

    kaggle_api = KaggleApiExtended()
    try:
        kaggle_api.authenticate()
    except SystemExit as e:
        raise RuntimeError(
            "Kaggle 認証に失敗しました。~/.kaggle/access_token を確認してください。"
        ) from e

    result = PushNotebookUseCase(cfg=cfg, platform_api=kaggle_api).execute()
    logging.getLogger(__name__).info(f"Notebook push 完了: notebook={result.notebook_path}")


def _run_inference(cfg: DictConfig) -> None:
    """推論 UseCase を実行する（Pipeline から呼ばれる）。"""
    from src.infrastructure.inference.lgbm_inferencer import LightGBMInferencer
    from src.infrastructure.repository.git import GitRepositoryImpl
    from src.usecase.inference.inference import InferenceUseCase
    from src.usecase.inference.inference_loader import load_inference_cfgs

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

    if usecase_name == "download_dataset":
        from src.usecase.data_acquisition.download_dataset import DownloadDatasetUseCase

        try:
            downloader = _resolve_downloader(cfg)
        except Exception:
            logger.error("ダウンローダーの初期化に失敗しました", exc_info=True)
            raise
        DownloadDatasetUseCase(downloader, logger).execute()  # ty:ignore[invalid-argument-type]

    elif usecase_name == "automatically_eda":
        from src.usecase.eda.automatically_eda import AutomaticallyEDAUseCase

        analyzers = _resolve_analyzers(cfg)
        AutomaticallyEDAUseCase(analyzers, logger).execute()  # ty:ignore[invalid-argument-type]

    elif usecase_name == "preprocess":
        _run_preprocess(cfg)

    elif usecase_name == "train":
        _run_train(cfg)

    elif usecase_name == "inference":
        _run_inference(cfg)

    elif usecase_name == "remote_train":
        _run_remote_train(cfg)

    elif usecase_name == "vertex_submit":
        _run_vertex_submit(cfg)

    elif usecase_name == "vertex_download":
        _run_vertex_download(cfg)

    elif usecase_name == "pipeline":
        from src.usecase.pipeline.pipeline import PipelineUseCase
        from src.usecase.pipeline.pipeline_loader import load_pipeline_recipe_cfg

        pipeline_cfg = load_pipeline_recipe_cfg(cfg, Path(_CONF_DIR))
        PipelineUseCase(
            run_preprocess=_run_preprocess,
            run_train=_run_train,
            run_inference=_run_inference,
            conf_dir=Path(_CONF_DIR),
            run_remote_train=_run_remote_train,
            run_vertex_submit=_run_vertex_submit,
            run_vertex_download=_run_vertex_download,
            run_update_source_dataset=_run_update_source_dataset_pipeline,
            run_push_notebook=_run_push_notebook_pipeline,
            run_download_dataset=_run_download,
        ).run(pipeline_cfg)
        logger.info(f"パイプライン完了[{pipeline_cfg.get('job_id', '?')}]")

    elif usecase_name == "push_notebook":
        from src.usecase.notebook.push_notebook import PushNotebookUseCase

        try:
            from kaggle.api.kaggle_api_extended import (
                KaggleApi as KaggleApiExtended,
            )
        except SystemExit as e:
            raise RuntimeError(
                "Kaggle 認証に失敗しました。~/.kaggle/access_token を確認してください。"
            ) from e

        kaggle_api = KaggleApiExtended()
        try:
            kaggle_api.authenticate()
        except SystemExit as e:
            raise RuntimeError(
                "Kaggle 認証に失敗しました。~/.kaggle/access_token を確認してください。"
            ) from e

        result = PushNotebookUseCase(cfg=cfg, platform_api=kaggle_api).execute()
        logger.info(f"Notebook push 完了: notebook={result.notebook_path}")

    elif usecase_name == "gradcam":
        from src.infrastructure.analyzer.gradcam_analyzer import GradCAMAnalyzerImpl
        from src.infrastructure.repository.git import GitRepositoryImpl
        from src.usecase.analysis.gradcam import GradCAMUseCase

        git_repo = GitRepositoryImpl()
        raw_cfg: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # ty:ignore[invalid-assignment]
        results = GradCAMUseCase(
            cfg=raw_cfg,
            analyzer=GradCAMAnalyzerImpl(),
            git_repo=git_repo,
        ).execute()
        logger.info(f"GradCAM 完了: {len(results)} 画像分析")

    elif usecase_name in ("create_source_dataset", "update_source_dataset"):
        from src.infrastructure.kaggle.source_dataset import KaggleSourceDatasetRepository

        try:
            from kaggle.api.kaggle_api_extended import (
                KaggleApi as KaggleApiExtended,
            )
        except SystemExit as e:
            raise RuntimeError(
                "Kaggle 認証に失敗しました。~/.kaggle/access_token を確認してください。"
            ) from e

        kaggle_api = KaggleApiExtended()
        try:
            kaggle_api.authenticate()
        except SystemExit as e:
            raise RuntimeError(
                "Kaggle 認証に失敗しました。~/.kaggle/access_token を確認してください。"
            ) from e

        repository = KaggleSourceDatasetRepository(kaggle_api=kaggle_api)

        if usecase_name == "create_source_dataset":
            from src.usecase.source_dataset.create_source_dataset import (
                CreateSourceDatasetUseCase,
            )

            CreateSourceDatasetUseCase(cfg=cfg, repository=repository).execute()
            logger.info("create_source_dataset 完了")
        else:
            from src.usecase.source_dataset.update_source_dataset import (
                UpdateSourceDatasetUseCase,
            )

            UpdateSourceDatasetUseCase(cfg=cfg, repository=repository).execute()
            logger.info("update_source_dataset 完了")

    else:
        supported = [
            "download_dataset",
            "automatically_eda",
            "preprocess",
            "train",
            "inference",
            "remote_train",
            "vertex_submit",
            "vertex_download",
            "pipeline",
            "push_notebook",
            "create_source_dataset",
            "update_source_dataset",
            "gradcam",
        ]
        raise ValueError(f"Unknown usecase: {usecase_name!r}. Supported: {supported}")


if __name__ == "__main__":
    main()
