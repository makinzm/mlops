"""
VertexAITrainUseCase — GCP Vertex AI でモデル学習を実行するユースケース。

処理フロー:
1. preprocess_output_dir を解決（"latest" → 最新タイムスタンプ）
2. src/ + conf/ を GCS にアップロード（コードはイメージに含めず GCS 経由で渡す）
3. preprocessed data を GCS にアップロード
4. Vertex AI CustomJob を送信（コンテナに GCS_CODE_URI / GCS_DATA_URI / GCS_MODEL_URI を渡す）
5. ジョブ完了をポーリング（失敗時は RuntimeError を送出）
6. GCS からモデル成果物をローカルにダウンロード
7. per-job .gitignore を配置
8. VertexTrainResult を返す

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

# プロジェクトルート（src/ の親ディレクトリ）
# src/usecase/training/vertex_train.py → .parent x4 → プロジェクトルート
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


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
        gcs_code_uri = f"{bucket_base}/jobs/{job_id}/{timestamp}/code"
        gcs_data_uri = f"{bucket_base}/jobs/{job_id}/{timestamp}/data"
        gcs_model_uri = f"{bucket_base}/jobs/{job_id}/{timestamp}/models"

        # 3. src/ + conf/ + scripts/ を GCS にアップロード
        #    Docker イメージにはコードを含めず、GCS 経由で渡す
        logger.info(f"Uploading code to {gcs_code_uri}")
        self._gcs.upload_dir(_PROJECT_ROOT / "src", f"{gcs_code_uri}/src")
        self._gcs.upload_dir(_PROJECT_ROOT / "conf", f"{gcs_code_uri}/conf")
        self._gcs.upload_dir(_PROJECT_ROOT / "scripts", f"{gcs_code_uri}/scripts")

        # 4. GCS にデータをアップロード
        logger.info(f"Uploading preprocessed data to {gcs_data_uri}")
        self._gcs.upload_dir(preprocess_dir, gcs_data_uri)

        # 5. Vertex AI ジョブを送信
        #    Kaggle プリビルトイメージを使う場合、command で:
        #    - Python SDK でコードを GCS から /app にダウンロード
        #      (gsutil は Kaggle コンテナ内で ADC を自動取得できないため使わない)
        #    - 不足 deps を pip install
        #    - entrypoint を実行
        env_vars: dict[str, str] = {
            "GCS_CODE_URI": gcs_code_uri,
            "GCS_DATA_URI": gcs_data_uri,
            "GCS_MODEL_URI": gcs_model_uri,
            "MLOPS_JOB_ID": job_id,
            "MLOPS_RECIPE": recipe,
            "MLOPS_COMPETITION": competition,
            "MLOPS_COMMIT_HASH": commit_hash,
            "PYTHONPATH": "/app",
        }
        # GCS からコードをダウンロードする Python ワンライナー
        # google-cloud-storage は Kaggle イメージに入っており、ADC も自動取得される
        gcs_download_py = (
            "import os; from google.cloud import storage; "
            f"uri='{gcs_code_uri}'; "
            "bkt,pfx=uri[5:].split('/',1); "
            "c=storage.Client(); "
            "[("
            "  os.makedirs(os.path.dirname(f'/app/{b.name[len(pfx)+1:]}'),exist_ok=True),"
            "  b.download_to_filename(f'/app/{b.name[len(pfx)+1:]}')"
            ") for b in c.list_blobs(bkt,prefix=pfx) if b.name[len(pfx)+1:]]"
        )
        bootstrap = (
            f'python -c "{gcs_download_py}"'
            " && pip install -q hydra-core omegaconf python-dotenv pydantic jinja2 mlflow"
            " && python /app/scripts/vertex_entrypoint.py"
        )
        # run_custom_job は送信 + 完了待機を1メソッドで行う
        result = self._vertex.run_custom_job(
            display_name=f"{job_id}-{timestamp}",
            container_uri=str(cfg.gcp.container_uri),
            command=["bash", "-c", bootstrap],
            args=[],
            machine_type=str(cfg.gcp.machine_type),
            env_vars=env_vars,
            service_account=str(cfg.gcp.service_account),
        )
        vertex_job_name = result.resource_name
        logger.info(f"Vertex AI job completed: {vertex_job_name} ({result.state})")

        if not result.is_succeeded:
            raise RuntimeError(
                f"Vertex AI job failed [{vertex_job_name}]: "
                f"state={result.state}, error={result.error_message}"
            )

        # 7. ローカルにモデル成果物をダウンロード
        job_dir = Path(str(cfg.output_dir)) / job_id
        local_model_dir = job_dir / timestamp
        local_model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading model artifacts from {gcs_model_uri} to {local_model_dir}")
        self._gcs.download_dir(gcs_model_uri, local_model_dir)

        # 8. per-job .gitignore を配置
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
