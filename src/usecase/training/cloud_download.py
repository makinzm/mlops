"""
CloudDownloadUseCase — クラウド学習ジョブの結果をダウンロードする。

処理フロー:
1. job_manifest.yaml を読み込む
2. クラウド学習ジョブのステータスを確認
3. SUCCEEDED ならばオブジェクトストレージからモデルをダウンロード
4. manifest を DOWNLOADED に更新
5. CloudDownloadResult を返す

前提:
  CloudSubmitUseCase が job_manifest.yaml を生成済みであること。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.domain.data.job_manifest import JobManifest
from src.domain.repository.object_storage import ObjectStorageRepository
from src.domain.repository.training_job import TrainingJobRepository

logger = logging.getLogger(__name__)


@dataclass
class CloudDownloadResult:
    """クラウドモデルダウンロードの結果。"""

    job_id: str
    local_model_dir: str
    manifest_path: str


class CloudDownloadUseCase:
    """クラウド学習ジョブの結果をオブジェクトストレージからダウンロードする。

    Args:
        manifest_path: job_manifest.yaml のパス。
        object_storage: ObjectStorageRepository の実装。
        training_job: TrainingJobRepository の実装。
        output_dir: モデル出力先のルートディレクトリ。
    """

    def __init__(
        self,
        manifest_path: Path,
        object_storage: ObjectStorageRepository,
        training_job: TrainingJobRepository,
        output_dir: Path,
    ) -> None:
        self._manifest_path = manifest_path
        self._object_storage = object_storage
        self._training_job = training_job
        self._output_dir = output_dir

    def execute(self) -> CloudDownloadResult:
        """ジョブステータスを確認し、成功ならモデルをダウンロードする。

        時間計算量: O(D) — D: ダウンロードファイル数
        空間計算量: O(D)

        Raises:
            RuntimeError: ジョブがまだ実行中 or 失敗した場合。
        """
        manifest = JobManifest.load(self._manifest_path)

        # 既にダウンロード済みならスキップ
        if manifest.status == "DOWNLOADED":
            logger.info(f"Job {manifest.job_id} is already downloaded: {manifest.local_model_dir}")
            return CloudDownloadResult(
                job_id=manifest.job_id,
                local_model_dir=manifest.local_model_dir or "",
                manifest_path=str(self._manifest_path),
            )

        # ジョブステータスを確認
        status_result = self._training_job.get_job_status(manifest.cloud_job_name)

        if status_result.state == "FAILED":
            # manifest も FAILED に更新
            manifest.status = "FAILED"
            manifest.completed_at = datetime.now().isoformat()
            manifest.save(self._manifest_path)
            raise RuntimeError(
                f"Cloud training job FAILED [{manifest.cloud_job_name}]: "
                f"{status_result.error_message}"
            )

        if status_result.state not in ("SUCCEEDED",):
            raise RuntimeError(
                f"Cloud training job is still {status_result.state} "
                f"[{manifest.cloud_job_name}]. "
                f"Please wait for completion."
            )

        # オブジェクトストレージからモデルをダウンロード
        local_model_dir = self._output_dir / manifest.job_id / manifest.timestamp
        local_model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading model from {manifest.gcs_model_uri} to {local_model_dir}")
        self._object_storage.download_dir(manifest.gcs_model_uri, local_model_dir)

        # manifest を更新
        manifest.status = "DOWNLOADED"
        manifest.local_model_dir = str(local_model_dir)
        manifest.completed_at = datetime.now().isoformat()
        manifest.save(self._manifest_path)
        logger.info(f"Model downloaded to {local_model_dir}")

        return CloudDownloadResult(
            job_id=manifest.job_id,
            local_model_dir=str(local_model_dir),
            manifest_path=str(self._manifest_path),
        )
