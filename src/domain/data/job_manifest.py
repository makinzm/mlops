"""
JobManifest — Vertex AI ジョブの submit/download 間でパス情報を共有する dataclass。

なぜここに定義するか:
  JobManifest は domain 層のデータ構造であり、UseCase 層が直接使う。
  Infrastructure 層には依存しない純粋なデータクラスとして定義する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class JobManifest:
    """Vertex AI ジョブのメタ情報。submit 時に YAML 保存し、download 時に読み込む。

    Attributes:
        job_id: ジョブ識別子（例: titanic_lgbm）
        competition: コンペティション名（例: titanic）
        recipe: 学習レシピ名（例: lgbm）
        timestamp: ジョブ投入時のタイムスタンプ（YYYYMMDDTHHMMSS）
        commit_hash: Git コミットハッシュ
        status: ジョブ状態 (SUBMITTED / SUCCEEDED / FAILED / DOWNLOADED)
        vertex_job_name: Vertex AI ジョブのリソース名
        gcs_code_uri: コードの GCS URI
        gcs_data_uri: データの GCS URI
        gcs_model_uri: モデルの GCS URI
        local_model_dir: ダウンロード後のローカルモデルディレクトリ（download 後に設定）
        submitted_at: ジョブ投入日時（ISO 8601）
        completed_at: ジョブ完了日時（ISO 8601、完了後に設定）
        cv_score: CV スコア（完了後に設定）
    """

    job_id: str
    competition: str
    recipe: str
    timestamp: str
    commit_hash: str
    status: str = "SUBMITTED"
    vertex_job_name: str = ""
    gcs_code_uri: str = ""
    gcs_data_uri: str = ""
    gcs_model_uri: str = ""
    local_model_dir: str | None = None
    submitted_at: str = ""
    completed_at: str | None = None
    cv_score: str | None = None

    # 有効なステータス値
    VALID_STATUSES: list[str] = field(
        default_factory=lambda: ["SUBMITTED", "SUCCEEDED", "FAILED", "DOWNLOADED"],
        init=False,
        repr=False,
    )

    def save(self, path: Path) -> None:
        """manifest を YAML ファイルに保存する。

        時間計算量: O(F) — F: フィールド数
        空間計算量: O(F)
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "job_id": self.job_id,
            "competition": self.competition,
            "recipe": self.recipe,
            "timestamp": self.timestamp,
            "commit_hash": self.commit_hash,
            "status": self.status,
            "vertex_job_name": self.vertex_job_name,
            "gcs_code_uri": self.gcs_code_uri,
            "gcs_data_uri": self.gcs_data_uri,
            "gcs_model_uri": self.gcs_model_uri,
            "local_model_dir": self.local_model_dir,
            "submitted_at": self.submitted_at,
            "completed_at": self.completed_at,
            "cv_score": self.cv_score,
        }
        path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))

    @classmethod
    def load(cls, path: Path) -> JobManifest:
        """YAML ファイルから manifest を読み込む。

        時間計算量: O(F) — F: フィールド数
        空間計算量: O(F)

        Raises:
            FileNotFoundError: ファイルが存在しない場合
        """
        data = yaml.safe_load(path.read_text())
        return cls(
            job_id=data["job_id"],
            competition=data["competition"],
            recipe=data["recipe"],
            timestamp=data["timestamp"],
            commit_hash=data["commit_hash"],
            status=data.get("status", "SUBMITTED"),
            vertex_job_name=data.get("vertex_job_name", ""),
            gcs_code_uri=data.get("gcs_code_uri", ""),
            gcs_data_uri=data.get("gcs_data_uri", ""),
            gcs_model_uri=data.get("gcs_model_uri", ""),
            local_model_dir=data.get("local_model_dir"),
            submitted_at=data.get("submitted_at", ""),
            completed_at=data.get("completed_at"),
            cv_score=data.get("cv_score"),
        )
