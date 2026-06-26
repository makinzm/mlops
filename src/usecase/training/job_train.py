"""
JobTrainUseCase — 学習ジョブを送信し完了まで待機してモデルを取得するユースケース。

処理フロー:
1. preprocess_output_dir を解決（"latest" → 最新タイムスタンプ）
2. src/ + conf/ をオブジェクトストレージにアップロード（コードはイメージに含めず経由で渡す）
3. preprocessed data をオブジェクトストレージにアップロード
4. 学習ジョブを送信（コンテナに GCS_CODE_URI / GCS_DATA_URI / GCS_MODEL_URI を渡す）
5. ジョブ完了をポーリング（失敗時は RuntimeError を送出）
6. オブジェクトストレージからモデル成果物をローカルにダウンロード
7. per-job .gitignore を配置
8. JobTrainResult を返す

出力ディレクトリ構造:
  models/{competition}/{job_id}/
    ├── .gitignore
    └── {YYYYMMDDTHHMMSS}/
        ├── （コンテナが生成した fold_N/ など）
        └── （オブジェクトストレージからダウンロードされた全成果物）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig

from src.domain.repository.git import GitRepository
from src.domain.repository.object_storage import ObjectStorageRepository
from src.domain.repository.training_job import TrainingJobRepository
from src.usecase._utils import resolve_latest_dir

logger = logging.getLogger(__name__)

_MODELS_DIR_GITIGNORE = """\
*
!.gitignore
!*.yaml
!*.md
!*/
"""


def _find_project_root() -> Path:
    """pyproject.toml を含むディレクトリを走査して project root を検出する。

    時間計算量: O(D) — D: ディレクトリ深度
    空間計算量: O(1)
    """
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # フォールバック: 従来の相対パス
    return Path(__file__).parent.parent.parent.parent


_PROJECT_ROOT = _find_project_root()


@dataclass
class JobTrainResult:
    """学習ジョブの同期実行結果。"""

    job_id: str
    timestamp: str
    commit_hash: str
    job_name: str
    gcs_data_uri: str
    gcs_model_uri: str
    local_model_dir: Path


class JobTrainUseCase:
    """オブジェクトストレージ + 学習基盤を使いジョブを同期実行してモデルを取得する。

    Args:
        cfg: Hydra DictConfig。以下のキーを使用する:
            - cfg.job_id: ジョブ識別子
            - cfg.preprocess_output_dir: 前処理済みデータのパス
            - cfg.output_dir: モデル出力先のルートディレクトリ
            - cfg.recipe: 使用する学習レシピ名
            - cfg.infra.project: インフラのプロジェクト ID
            - cfg.infra.staging_bucket: オブジェクトストレージバケット URI
            - cfg.infra.container_uri: Docker イメージ URI
            - cfg.infra.machine_type: マシンタイプ
            - cfg.infra.service_account: 学習 SA のメールアドレス
        object_storage: ObjectStorageRepository の実装。
        training_job: TrainingJobRepository の実装。
        git_repo: GitRepository の実装。
    """

    def __init__(
        self,
        cfg: DictConfig,
        object_storage: ObjectStorageRepository,
        training_job: TrainingJobRepository,
        git_repo: GitRepository,
    ) -> None:
        self._cfg = cfg
        self._object_storage = object_storage
        self._training_job = training_job
        self._git_repo = git_repo

    def execute(self) -> JobTrainResult:
        cfg = self._cfg
        job_id = str(cfg.job_id)
        recipe = str(cfg.get("recipe", "base"))
        competition = str(cfg.competition.name)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        commit_hash = self._git_repo.get_commit_hash()

        # 1. preprocess_output_dir の "latest" 解決
        preprocess_dir = resolve_latest_dir(str(cfg.preprocess_output_dir))

        # 2. オブジェクトストレージ URI を確定
        infra_cfg = cfg.infra
        bucket_base = str(infra_cfg.staging_bucket).rstrip("/")
        gcs_code_uri = f"{bucket_base}/jobs/{job_id}/{timestamp}/code"
        gcs_data_uri = f"{bucket_base}/jobs/{job_id}/{timestamp}/data"
        gcs_model_uri = f"{bucket_base}/jobs/{job_id}/{timestamp}/models"

        # 3. src/ + conf/ + scripts/ をオブジェクトストレージにアップロード
        #    Docker イメージにはコードを含めず、オブジェクトストレージ経由で渡す
        logger.info(f"Uploading code to {gcs_code_uri}")
        self._object_storage.upload_dir(_PROJECT_ROOT / "src", f"{gcs_code_uri}/src")
        self._object_storage.upload_dir(_PROJECT_ROOT / "conf", f"{gcs_code_uri}/conf")
        self._object_storage.upload_dir(_PROJECT_ROOT / "scripts", f"{gcs_code_uri}/scripts")

        # 4. オブジェクトストレージにデータをアップロード
        logger.info(f"Uploading preprocessed data to {gcs_data_uri}")
        self._object_storage.upload_dir(preprocess_dir, gcs_data_uri)

        # 5. 学習ジョブを送信
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
        # bootstrap コマンド構築は Infrastructure 層の責務
        command = self._training_job.build_bootstrap_command(gcs_code_uri)
        # run_custom_job は送信 + 完了待機を1メソッドで行う
        result = self._training_job.run_custom_job(
            display_name=f"{job_id}-{timestamp}",
            container_uri=str(infra_cfg.container_uri),
            command=command,
            args=[],
            machine_type=str(infra_cfg.machine_type),
            env_vars=env_vars,
            service_account=str(infra_cfg.service_account),
        )
        job_name = result.resource_name
        logger.info(f"Training job completed: {job_name} ({result.state})")

        if not result.is_succeeded:
            raise RuntimeError(
                f"Training job failed [{job_name}]: "
                f"state={result.state}, error={result.error_message}"
            )

        # 6. ローカルにモデル成果物をダウンロード
        job_dir = Path(str(cfg.output_dir)) / job_id
        local_model_dir = job_dir / timestamp
        local_model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading model artifacts from {gcs_model_uri} to {local_model_dir}")
        self._object_storage.download_dir(gcs_model_uri, local_model_dir)

        # 7. per-job .gitignore を配置
        gitignore_path = job_dir / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text(_MODELS_DIR_GITIGNORE)

        return JobTrainResult(
            job_id=job_id,
            timestamp=timestamp,
            commit_hash=commit_hash,
            job_name=job_name,
            gcs_data_uri=gcs_data_uri,
            gcs_model_uri=gcs_model_uri,
            local_model_dir=local_model_dir,
        )
