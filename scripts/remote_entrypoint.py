"""
Vertex AI カスタムトレーニングコンテナのエントリーポイント。

前提:
  ジョブ起動時の command で src/ + conf/ は /app/ に配置済み。
  このスクリプトはデータのダウンロード → 学習 → モデルのアップロードを担う。

処理フロー:
1. GCS_DATA_URI から /tmp/mlops/data/ にデータをダウンロード
2. 環境変数 MLOPS_DATA_DIR / MLOPS_MODEL_DIR を設定
3. python -m src usecase=train を実行
4. /tmp/mlops/models/ を GCS_MODEL_URI にアップロード

環境変数（Vertex AI ジョブから注入）:
  GCS_DATA_URI     : 前処理済みデータの GCS URI
  GCS_MODEL_URI    : モデル出力先の GCS URI
  MLOPS_JOB_ID     : ジョブ識別子
  MLOPS_RECIPE     : 学習レシピ名（例: lgbm）
  MLOPS_COMPETITION: コンペティション名（例: titanic）
  MLOPS_COMMIT_HASH: Git コミットハッシュ
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """gs://bucket/prefix → (bucket, prefix)"""
    assert gcs_uri.startswith("gs://"), f"Invalid GCS URI: {gcs_uri}"
    path = gcs_uri[5:]
    parts = path.split("/", 1)
    return parts[0], (parts[1].rstrip("/") if len(parts) > 1 else "")


def download_from_gcs(gcs_uri: str, local_dir: Path) -> None:
    """GCS URI 以下のファイルを local_dir にダウンロードする。"""
    bucket_name, prefix = _parse_gcs_uri(gcs_uri)
    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    logger.info(f"Downloading {len(blobs)} files from {gcs_uri}")
    for blob in blobs:
        relative = blob.name[len(prefix) :].lstrip("/") if prefix else blob.name
        if not relative:
            continue
        local_file = local_dir / relative
        local_file.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_file))
    logger.info(f"Download complete: {local_dir}")


def upload_to_gcs(local_dir: Path, gcs_uri: str) -> None:
    """local_dir 以下の全ファイルを GCS URI にアップロードする。"""
    bucket_name, prefix = _parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    logger.info(f"Uploading {len(files)} files to {gcs_uri}")
    for local_file in files:
        relative = local_file.relative_to(local_dir)
        blob_name = f"{prefix}/{relative}" if prefix else str(relative)
        bucket.blob(blob_name).upload_from_filename(str(local_file))
    logger.info(f"Upload complete: {gcs_uri}")


def main() -> None:
    gcs_data_uri = os.environ["GCS_DATA_URI"]
    gcs_model_uri = os.environ["GCS_MODEL_URI"]
    job_id = os.environ.get("MLOPS_JOB_ID", "vertex_job")
    recipe = os.environ.get("MLOPS_RECIPE", "lgbm")
    competition = os.environ.get("MLOPS_COMPETITION", "titanic")

    data_dir = Path("/tmp/mlops/data")
    model_dir = Path("/tmp/mlops/models")
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. GCS からデータをダウンロード
    logger.info("=== Step 1: Downloading data from GCS ===")
    download_from_gcs(gcs_data_uri, data_dir)

    # 2. 環境変数を設定（Hydra の ${oc.env:...} で読まれる）
    os.environ["MLOPS_DATA_DIR"] = str(data_dir)
    os.environ["MLOPS_MODEL_DIR"] = str(model_dir)

    # 3. 学習を実行
    logger.info("=== Step 2: Running training ===")
    cmd = [
        "python",
        "-m",
        "src",
        "usecase=train",
        f"recipe={recipe}",
        f"job_id={job_id}",
        f"competition={competition}",
    ]
    logger.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/app", check=False)
    if result.returncode != 0:
        logger.error(f"Training failed with return code {result.returncode}")
        sys.exit(result.returncode)

    # 4. GCS にモデルをアップロード
    #    TrainUseCase は model_dir/{job_id}/{timestamp}/fold_0/model.lgbm の構造で出力する。
    #    UseCase 側は GCS の内容を models/titanic/titanic_lgbm/{local_ts}/ にダウンロードするので、
    #    fold_0/ が直下に来るよう、最も深いタイムスタンプディレクトリを見つけてアップロードする。
    logger.info("=== Step 3: Uploading models to GCS ===")
    import glob as _glob

    fold_markers = _glob.glob(str(model_dir / "**" / "fold_0" / "model.lgbm"), recursive=True)
    if fold_markers:
        # fold_0/model.lgbm の 2 つ上がタイムスタンプディレクトリ
        actual_model_dir = Path(fold_markers[0]).parent.parent
        logger.info(f"Found model output at: {actual_model_dir}")
        upload_to_gcs(actual_model_dir, gcs_model_uri)
    else:
        logger.warning("fold_0/model.lgbm が見つかりません。model_dir 全体をアップロードします。")
        upload_to_gcs(model_dir, gcs_model_uri)
    logger.info("=== Training pipeline complete ===")


if __name__ == "__main__":
    main()
