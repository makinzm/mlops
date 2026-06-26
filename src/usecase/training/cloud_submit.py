"""
CloudSubmitUseCase — クラウド環境にジョブを非同期送信し、即座に終了する。

処理フロー:
1. preprocess_output_dir を解決（"latest" -> 最新タイムスタンプ）
2. src/ + conf/ + scripts/ をオブジェクトストレージにアップロード
3. preprocessed data をオブジェクトストレージにアップロード
4. submit_custom_job (sync=False) でジョブ送信
5. job_manifest.yaml を保存
6. CloudSubmitResult を返す（ローカルプロセスは即終了可能）

出力:
  models/{competition}/{job_id}/
    ├── .gitignore
    └── job_manifest.yaml

CloudTrainUseCase（同期版）との違い:
- submit_custom_job を使い、ジョブ完了を待たない
- モデルのダウンロードは行わない（CloudDownloadUseCase が担う）
- 通知用環境変数をコンテナに渡す
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig

from src.domain.data.job_manifest import JobManifest
from src.domain.repository.git import GitRepository
from src.domain.repository.object_storage import ObjectStorageRepository
from src.domain.repository.training_job import TrainingJobRepository
from src.usecase._utils import resolve_latest_dir

logger = logging.getLogger(__name__)

_HISTORY_DIR_GITIGNORE = """\
# manifest にはクラウドプロジェクト ID 等が含まれるため git 追跡しない。
# ディレクトリ構造（.gitkeep）のみ残し、試行回数を確認できるようにする。
*
!.gitignore
!.gitkeep
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
    return Path(__file__).parent.parent.parent.parent


_PROJECT_ROOT = _find_project_root()


@dataclass
class CloudSubmitResult:
    """クラウドジョブ送信の結果。"""

    job_id: str
    timestamp: str
    commit_hash: str
    cloud_job_name: str
    manifest_path: str
    console_url: str


class CloudSubmitUseCase:
    """クラウド環境にジョブを非同期送信し、manifest を保存して即座に終了する。

    Args:
        cfg: Hydra DictConfig。以下のキーを使用する:
            - cfg.job_id: ジョブ識別子
            - cfg.preprocess_output_dir: 前処理済みデータのパス
            - cfg.output_dir: モデル出力先のルートディレクトリ
            - cfg.recipe: 使用する学習レシピ名
            - cfg.cloud.*: クラウド設定
            - cfg.notification: 通知設定（オプション）
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

    def execute(self) -> CloudSubmitResult:
        """ジョブを非同期送信し、manifest を保存して結果を返す。

        時間計算量: O(U) — U: アップロードファイル数
        空間計算量: O(1)（manifest YAML のみ）
        """
        cfg = self._cfg
        job_id = str(cfg.job_id)
        recipe = str(cfg.get("recipe", "base"))
        competition = str(cfg.competition.name)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        commit_hash = self._git_repo.get_commit_hash()

        # 1. preprocess_output_dir の "latest" 解決
        preprocess_dir = resolve_latest_dir(str(cfg.preprocess_output_dir))

        # 2. オブジェクトストレージ URI を確定
        cloud_cfg = cfg.cloud
        bucket_base = str(cloud_cfg.staging_bucket).rstrip("/")
        gcs_code_uri = f"{bucket_base}/jobs/{job_id}/{timestamp}/code"
        gcs_data_uri = f"{bucket_base}/jobs/{job_id}/{timestamp}/data"
        gcs_model_uri = f"{bucket_base}/jobs/{job_id}/{timestamp}/models"

        # 3. src/ + conf/ + scripts/ をアップロード
        logger.info(f"Uploading code to {gcs_code_uri}")
        self._object_storage.upload_dir(_PROJECT_ROOT / "src", f"{gcs_code_uri}/src")
        self._object_storage.upload_dir(_PROJECT_ROOT / "conf", f"{gcs_code_uri}/conf")
        self._object_storage.upload_dir(_PROJECT_ROOT / "scripts", f"{gcs_code_uri}/scripts")

        # 4. データをアップロード
        logger.info(f"Uploading preprocessed data to {gcs_data_uri}")
        self._object_storage.upload_dir(preprocess_dir, gcs_data_uri)

        # 5. 環境変数の構築（通知設定を含む）
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
        # 通知設定をコンテナ環境変数に追加
        notification_cfg = cfg.get("notification")
        if notification_cfg:
            slack_cfg = notification_cfg.get("slack")
            if slack_cfg and slack_cfg.get("webhook_url"):
                env_vars["SLACK_WEBHOOK_URL"] = str(slack_cfg.webhook_url)
            email_cfg = notification_cfg.get("email")
            if email_cfg:
                for key in (
                    "smtp_host",
                    "smtp_port",
                    "sender",
                    "recipient",
                    "username",
                    "password",
                ):
                    value = email_cfg.get(key)
                    if value is not None:
                        env_vars[f"SMTP_{key.upper()}"] = str(value)

        # 6. ジョブ送信（非同期）
        #    bootstrap コマンド構築は Infrastructure 層の責務
        command = self._training_job.build_bootstrap_command(gcs_code_uri)
        result = self._training_job.submit_custom_job(
            display_name=f"{job_id}-{timestamp}",
            container_uri=str(cloud_cfg.container_uri),
            command=command,
            args=[],
            machine_type=str(cloud_cfg.machine_type),
            env_vars=env_vars,
            service_account=str(cloud_cfg.service_account),
        )
        cloud_job_name = result.resource_name

        # 7. job_manifest.yaml を cloud_jobs_history/ に保存
        #    models/ はモデル成果物専用。ジョブ管理情報は分離する。
        history_base = str(cfg.get("cloud_jobs_history_dir", "cloud_jobs_history"))
        if not Path(history_base).is_absolute():
            history_base = str(_PROJECT_ROOT / history_base)
        history_dir = Path(history_base) / competition / job_id / timestamp
        manifest = JobManifest(
            job_id=job_id,
            competition=competition,
            recipe=recipe,
            timestamp=timestamp,
            commit_hash=commit_hash,
            status="SUBMITTED",
            cloud_job_name=cloud_job_name,
            gcs_code_uri=gcs_code_uri,
            gcs_data_uri=gcs_data_uri,
            gcs_model_uri=gcs_model_uri,
            submitted_at=datetime.now().isoformat(),
        )
        manifest_path = history_dir / "job_manifest.yaml"
        manifest.save(manifest_path)
        logger.info(f"Job manifest saved: {manifest_path}")

        # 8. per-directory .gitignore + .gitkeep を配置
        gitignore_path = history_dir / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text(_HISTORY_DIR_GITIGNORE)
        gitkeep_path = history_dir / ".gitkeep"
        if not gitkeep_path.exists():
            gitkeep_path.touch()

        # Console URL を TrainingJobRepository 経由で生成
        console_url = self._training_job.build_console_url(cloud_job_name)

        logger.info(
            f"Cloud job submitted: {cloud_job_name}\n"
            f"  Console: {console_url}\n"
            f"  manifest: {manifest_path}\n"
            f"  Download: uv run python -m src usecase=cloud_download"
        )

        return CloudSubmitResult(
            job_id=job_id,
            timestamp=timestamp,
            commit_hash=commit_hash,
            cloud_job_name=cloud_job_name,
            manifest_path=str(manifest_path),
            console_url=console_url,
        )
