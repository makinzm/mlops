"""
JobManifest の単体テスト。

なぜこのテストが必要か:
  - JobManifest は vertex_submit と vertex_download を繋ぐ重要なデータ構造。
  - YAML への保存・読み込みのラウンドトリップが正しく動作することを保証する。
  - ステータス遷移（SUBMITTED → SUCCEEDED → DOWNLOADED）が正しく記録されることを保証する。
"""

from __future__ import annotations

from pathlib import Path

from src.domain.data.job_manifest import JobManifest


def _make_manifest() -> JobManifest:
    """テスト用の JobManifest を生成する。"""
    return JobManifest(
        job_id="titanic_lgbm",
        competition="titanic",
        recipe="lgbm",
        timestamp="20260325T143000",
        commit_hash="a" * 40,
        status="SUBMITTED",
        remote_job_name="projects/123/locations/asia-northeast1/customJobs/789",
        gcs_code_uri="gs://bucket/jobs/titanic_lgbm/20260325T143000/code",
        gcs_data_uri="gs://bucket/jobs/titanic_lgbm/20260325T143000/data",
        gcs_model_uri="gs://bucket/jobs/titanic_lgbm/20260325T143000/models",
        submitted_at="2026-03-25T14:30:00",
    )


class TestJobManifestSaveLoad:
    """JobManifest の save/load ラウンドトリップテスト。"""

    def test_save_creates_yaml_file(self, tmp_path: Path) -> None:
        """save() が YAML ファイルを作成すること。"""
        manifest = _make_manifest()
        path = tmp_path / "models" / "titanic" / "titanic_lgbm" / "job_manifest.yaml"
        manifest.save(path)
        assert path.exists()

    def test_load_restores_manifest(self, tmp_path: Path) -> None:
        """save → load のラウンドトリップで全フィールドが復元されること。"""
        manifest = _make_manifest()
        path = tmp_path / "job_manifest.yaml"
        manifest.save(path)

        loaded = JobManifest.load(path)
        assert loaded.job_id == manifest.job_id
        assert loaded.competition == manifest.competition
        assert loaded.recipe == manifest.recipe
        assert loaded.timestamp == manifest.timestamp
        assert loaded.commit_hash == manifest.commit_hash
        assert loaded.status == manifest.status
        assert loaded.remote_job_name == manifest.remote_job_name
        assert loaded.gcs_code_uri == manifest.gcs_code_uri
        assert loaded.gcs_data_uri == manifest.gcs_data_uri
        assert loaded.gcs_model_uri == manifest.gcs_model_uri
        assert loaded.submitted_at == manifest.submitted_at
        assert loaded.local_model_dir is None
        assert loaded.completed_at is None

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        """save() がネストされたディレクトリを自動作成すること。"""
        manifest = _make_manifest()
        path = tmp_path / "deep" / "nested" / "dir" / "job_manifest.yaml"
        manifest.save(path)
        assert path.exists()

    def test_status_update_persists(self, tmp_path: Path) -> None:
        """ステータス更新が save/load で保持されること。"""
        manifest = _make_manifest()
        path = tmp_path / "job_manifest.yaml"
        manifest.status = "SUCCEEDED"
        manifest.completed_at = "2026-03-25T16:30:00"
        manifest.cv_score = "0.832"
        manifest.save(path)

        loaded = JobManifest.load(path)
        assert loaded.status == "SUCCEEDED"
        assert loaded.completed_at == "2026-03-25T16:30:00"
        assert loaded.cv_score == "0.832"

    def test_downloaded_status_with_local_model_dir(self, tmp_path: Path) -> None:
        """DOWNLOADED ステータスで local_model_dir が保持されること。"""
        manifest = _make_manifest()
        path = tmp_path / "job_manifest.yaml"
        manifest.status = "DOWNLOADED"
        manifest.local_model_dir = "/home/user/models/titanic/titanic_lgbm/20260325T143000"
        manifest.save(path)

        loaded = JobManifest.load(path)
        assert loaded.status == "DOWNLOADED"
        assert loaded.local_model_dir == "/home/user/models/titanic/titanic_lgbm/20260325T143000"


class TestJobManifestDefaults:
    """JobManifest のデフォルト値テスト。"""

    def test_default_status_is_submitted(self) -> None:
        """デフォルトの status が SUBMITTED であること。"""
        manifest = JobManifest(
            job_id="test",
            competition="test",
            recipe="test",
            timestamp="20260325T000000",
            commit_hash="x" * 40,
        )
        assert manifest.status == "SUBMITTED"

    def test_default_optional_fields_are_none(self) -> None:
        """オプショナルフィールドのデフォルトが None であること。"""
        manifest = JobManifest(
            job_id="test",
            competition="test",
            recipe="test",
            timestamp="20260325T000000",
            commit_hash="x" * 40,
        )
        assert manifest.local_model_dir is None
        assert manifest.completed_at is None
        assert manifest.cv_score is None
