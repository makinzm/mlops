"""
VertexAITrainUseCase — GCP Vertex AI でモデル学習を実行するユースケース。

処理フロー:
1. preprocess_output_dir を解決（"latest" → 最新タイムスタンプ）
2. preprocessed data を GCS にアップロード
3. Vertex AI CustomJob を送信（コンテナに GCS_DATA_URI / GCS_MODEL_URI を渡す）
4. ジョブ完了をポーリング（失敗時は RuntimeError を送出）
5. GCS からモデル成果物をローカルにダウンロード
6. per-job .gitignore を配置
7. VertexTrainResult を返す

出力ディレクトリ構造:
  models/{competition}/{job_id}/
    ├── .gitignore
    └── {YYYYMMDDTHHMMSS}/
        ├── （Vertex AI コンテナが生成した fold_N/ など）
        └── （GCS からダウンロードされた全成果物）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig

from src.domain.repository.gcs import GCSRepository
from src.domain.repository.git import GitRepository
from src.domain.repository.vertex_ai import VertexAIRepository
from src.usecase._utils import resolve_latest_dir

logger = logging.getLogger(__name__)

_MODELS_DIR_GITIGNORE = """\
*
!.gitignore
!*.yaml
!*.md
!*/
"""


@dataclass
class VertexTrainResult:
    """Vertex AI トレーニングジョブの実行結果。"""

    job_id: str
    timestamp: str
    commit_hash: str
    vertex_job_name: str
    gcs_data_uri: str
    gcs_model_uri: str
    local_model_dir: Path


class VertexAITrainUseCase:
    """GCS + Vertex AI を使いリモートでモデル学習を実行する。

    Args:
        cfg: Hydra DictConfig。以下のキーを使用する:
            - cfg.job_id: ジョブ識別子
            - cfg.preprocess_output_dir: 前処理済みデータのパス
            - cfg.output_dir: モデル出力先のルートディレクトリ
            - cfg.recipe: 使用する学習レシピ名
            - cfg.gcp.project: GCP プロジェクト ID
            - cfg.gcp.staging_bucket: GCS バケット URI
            - cfg.gcp.container_uri: Docker イメージ URI
            - cfg.gcp.machine_type: Vertex AI マシンタイプ
            - cfg.gcp.service_account: 学習 SA のメールアドレス
        gcs: GCSRepository の実装。
        vertex: VertexAIRepository の実装。
        git_repo: GitRepository の実装。
    """

    def __init__(
        self,
        cfg: DictConfig,
        gcs: GCSRepository,
        vertex: VertexAIRepository,
        git_repo: GitRepository,
    ) -> None:
        self._cfg = cfg
        self._gcs = gcs
        self._vertex = vertex
        self._git_repo = git_repo

    def execute(self) -> VertexTrainResult:
        cfg = self._cfg
        job_id = str(cfg.job_id)
        recipe = str(cfg.get("recipe", "lgbm"))
        competition = str(cfg.competition.name)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        commit_hash = self._git_repo.get_commit_hash()

        # 1. preprocess_output_dir の "latest" 解決
        preprocess_dir = resolve_latest_dir(str(cfg.preprocess_output_dir))

        # 2. GCS URI を確定
        bucket_base = str(cfg.gcp.staging_bucket).rstrip("/")
        gcs_data_uri = f"{bucket_base}/jobs/{job_id}/{timestamp}/data"
        gcs_model_uri = f"{bucket_base}/jobs/{job_id}/{timestamp}/models"

        # 3. GCS にデータをアップロード
        logger.info(f"Uploading preprocessed data to {gcs_data_uri}")
        self._gcs.upload_dir(preprocess_dir, gcs_data_uri)

        # 4. Vertex AI ジョブを送信
        env_vars: dict[str, str] = {
            "GCS_DATA_URI": gcs_data_uri,
            "GCS_MODEL_URI": gcs_model_uri,
            "MLOPS_JOB_ID": job_id,
            "MLOPS_RECIPE": recipe,
            "MLOPS_COMPETITION": competition,
            "MLOPS_COMMIT_HASH": commit_hash,
        }
        vertex_job_name = self._vertex.submit_custom_job(
            display_name=f"{job_id}-{timestamp}",
            container_uri=str(cfg.gcp.container_uri),
            args=[],
            machine_type=str(cfg.gcp.machine_type),
            env_vars=env_vars,
            service_account=str(cfg.gcp.service_account),
        )
        logger.info(f"Submitted Vertex AI job: {vertex_job_name}")

        # 5. ジョブ完了を待機
        status = self._vertex.wait_for_job(vertex_job_name)
        if not status.is_succeeded:
            raise RuntimeError(
                f"Vertex AI job failed [{vertex_job_name}]: "
                f"state={status.state}, error={status.error_message}"
            )

        # 6. ローカルにモデル成果物をダウンロード
        job_dir = Path(str(cfg.output_dir)) / job_id
        local_model_dir = job_dir / timestamp
        local_model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading model artifacts from {gcs_model_uri} to {local_model_dir}")
        self._gcs.download_dir(gcs_model_uri, local_model_dir)

        # 7. per-job .gitignore を配置
        gitignore_path = job_dir / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text(_MODELS_DIR_GITIGNORE)

        return VertexTrainResult(
            job_id=job_id,
            timestamp=timestamp,
            commit_hash=commit_hash,
            vertex_job_name=vertex_job_name,
            gcs_data_uri=gcs_data_uri,
            gcs_model_uri=gcs_model_uri,
            local_model_dir=local_model_dir,
        )
