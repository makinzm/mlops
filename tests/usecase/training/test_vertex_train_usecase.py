"""
VertexAITrainUseCase の単体テスト。

なぜこのテストが必要か:
  - VertexAITrainUseCase は GCSRepository と VertexAIRepository を受け取り、
    コード + データの GCS アップロード → Vertex AI ジョブ送信 → 完了待機 →
    モデルの GCS ダウンロード の一連の流れを担う。
  - 各リポジトリの Mock を使い、GCP 呼び出しなしで UseCase ロジックをテストする。
  - .gitignore 配置、VertexTrainResult の正しい構造を保証する。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig, OmegaConf

from src.domain.repository.vertex_ai import VertexJobStatus
from src.usecase.training.vertex_train import VertexAITrainUseCase, VertexTrainResult

_FAKE_COMMIT = "b" * 40
_FAKE_JOB_NAME = "projects/123/locations/asia-northeast1/customJobs/789"


def _make_cfg(tmp_path: Path) -> DictConfig:
    """VertexAITrainUseCase 用の DictConfig を生成する。"""
    processed_dir = tmp_path / "processed" / "titanic_preprocess" / "20260315T120000" / "train_out"
    processed_dir.mkdir(parents=True)
    (processed_dir / "fold_0").mkdir()
    (processed_dir / "fold_0" / "train.parquet").write_bytes(b"data")

    return OmegaConf.create(
        {
            "job_id": "titanic_lgbm",
            "competition": {"name": "titanic"},
            "preprocess_output_dir": str(processed_dir),
            "recipe": "lgbm",
            "output_dir": str(tmp_path / "models" / "titanic"),
            "seed": 42,
            "gcp": {
                "project": "test-project",
                "region": "asia-northeast1",
                "staging_bucket": "gs://test-bucket",
                "container_uri": "gcr.io/test/training:latest",
                "machine_type": "n1-standard-4",
                "service_account": "sa@test.iam.gserviceaccount.com",
            },
        }
    )


def _make_mock_gcs() -> MagicMock:
    mock = MagicMock()
    mock.upload_dir.return_value = None
    mock.download_dir.return_value = None
    return mock


def _make_mock_vertex() -> MagicMock:
    mock = MagicMock()
    mock.submit_custom_job.return_value = _FAKE_JOB_NAME
    mock.wait_for_job.return_value = VertexJobStatus(state="SUCCEEDED")
    return mock


def _make_mock_git_repo() -> MagicMock:
    mock = MagicMock()
    mock.get_commit_hash.return_value = _FAKE_COMMIT
    return mock


class TestVertexAITrainUseCaseExecute:
    """VertexAITrainUseCase.execute() のテスト。"""

    def test_returns_vertex_train_result(self, tmp_path: Path) -> None:
        """execute() が VertexTrainResult を返すこと。"""
        cfg = _make_cfg(tmp_path)
        usecase = VertexAITrainUseCase(
            cfg=cfg,
            gcs=_make_mock_gcs(),
            vertex=_make_mock_vertex(),
            git_repo=_make_mock_git_repo(),
        )
        result = usecase.execute()
        assert isinstance(result, VertexTrainResult)

    def test_uploads_code_and_data_to_gcs(self, tmp_path: Path) -> None:
        """src/ + conf/ + preprocessed data が GCS にアップロードされること。"""
        cfg = _make_cfg(tmp_path)
        mock_gcs = _make_mock_gcs()
        usecase = VertexAITrainUseCase(
            cfg=cfg, gcs=mock_gcs, vertex=_make_mock_vertex(), git_repo=_make_mock_git_repo()
        )
        usecase.execute()

        # upload_dir が 3 回呼ばれていること（src, conf, data）
        assert mock_gcs.upload_dir.call_count == 3
        upload_uris = [call.args[1] for call in mock_gcs.upload_dir.call_args_list]
        assert any("/code/src" in uri for uri in upload_uris)
        assert any("/code/conf" in uri for uri in upload_uris)
        assert any("/data" in uri for uri in upload_uris)

    def test_submits_vertex_job_with_correct_params(self, tmp_path: Path) -> None:
        """Vertex AI ジョブが正しいコンテナ URI とマシンタイプで送信されること。"""
        cfg = _make_cfg(tmp_path)
        mock_vertex = _make_mock_vertex()
        usecase = VertexAITrainUseCase(
            cfg=cfg, gcs=_make_mock_gcs(), vertex=mock_vertex, git_repo=_make_mock_git_repo()
        )
        usecase.execute()

        submit_call = mock_vertex.submit_custom_job.call_args
        assert submit_call.kwargs["container_uri"] == "gcr.io/test/training:latest"
        assert submit_call.kwargs["machine_type"] == "n1-standard-4"
        assert submit_call.kwargs["service_account"] == "sa@test.iam.gserviceaccount.com"

    def test_waits_for_job_completion(self, tmp_path: Path) -> None:
        """ジョブ完了まで wait_for_job が呼ばれること。"""
        cfg = _make_cfg(tmp_path)
        mock_vertex = _make_mock_vertex()
        usecase = VertexAITrainUseCase(
            cfg=cfg, gcs=_make_mock_gcs(), vertex=mock_vertex, git_repo=_make_mock_git_repo()
        )
        usecase.execute()

        mock_vertex.wait_for_job.assert_called_once_with(_FAKE_JOB_NAME)

    def test_downloads_model_artifacts_from_gcs(self, tmp_path: Path) -> None:
        """モデル成果物が GCS からローカルにダウンロードされること。"""
        cfg = _make_cfg(tmp_path)
        mock_gcs = _make_mock_gcs()
        usecase = VertexAITrainUseCase(
            cfg=cfg, gcs=mock_gcs, vertex=_make_mock_vertex(), git_repo=_make_mock_git_repo()
        )
        result = usecase.execute()

        mock_gcs.download_dir.assert_called_once()
        download_call = mock_gcs.download_dir.call_args
        gcs_uri: str = download_call.args[0]
        local_dir: Path = download_call.args[1]
        assert gcs_uri.startswith("gs://test-bucket/")
        assert result.local_model_dir == local_dir

    def test_creates_gitignore_in_job_dir(self, tmp_path: Path) -> None:
        """job_id ディレクトリに .gitignore が作成されること。"""
        cfg = _make_cfg(tmp_path)
        usecase = VertexAITrainUseCase(
            cfg=cfg,
            gcs=_make_mock_gcs(),
            vertex=_make_mock_vertex(),
            git_repo=_make_mock_git_repo(),
        )
        usecase.execute()

        job_dir = Path(str(cfg.output_dir)) / str(cfg.job_id)
        assert (job_dir / ".gitignore").exists()

    def test_raises_on_job_failure(self, tmp_path: Path) -> None:
        """Vertex AI ジョブが FAILED の場合に RuntimeError が送出されること。"""
        cfg = _make_cfg(tmp_path)
        mock_vertex = _make_mock_vertex()
        mock_vertex.wait_for_job.return_value = VertexJobStatus(
            state="FAILED", error_message="OOM error"
        )
        usecase = VertexAITrainUseCase(
            cfg=cfg, gcs=_make_mock_gcs(), vertex=mock_vertex, git_repo=_make_mock_git_repo()
        )

        with pytest.raises(RuntimeError, match="OOM error"):
            usecase.execute()

    def test_result_contains_commit_hash(self, tmp_path: Path) -> None:
        """VertexTrainResult に commit_hash が記録されること。"""
        cfg = _make_cfg(tmp_path)
        usecase = VertexAITrainUseCase(
            cfg=cfg,
            gcs=_make_mock_gcs(),
            vertex=_make_mock_vertex(),
            git_repo=_make_mock_git_repo(),
        )
        result = usecase.execute()
        assert result.commit_hash == _FAKE_COMMIT

    def test_result_contains_gcs_uris(self, tmp_path: Path) -> None:
        """VertexTrainResult に GCS URI が記録されること。"""
        cfg = _make_cfg(tmp_path)
        usecase = VertexAITrainUseCase(
            cfg=cfg,
            gcs=_make_mock_gcs(),
            vertex=_make_mock_vertex(),
            git_repo=_make_mock_git_repo(),
        )
        result = usecase.execute()
        assert result.gcs_data_uri.startswith("gs://")
        assert result.gcs_model_uri.startswith("gs://")

    def test_env_vars_contain_gcs_code_uri(self, tmp_path: Path) -> None:
        """Vertex AI ジョブの env_vars に GCS_CODE_URI が含まれること。"""
        cfg = _make_cfg(tmp_path)
        mock_vertex = _make_mock_vertex()
        usecase = VertexAITrainUseCase(
            cfg=cfg, gcs=_make_mock_gcs(), vertex=mock_vertex, git_repo=_make_mock_git_repo()
        )
        usecase.execute()

        submit_call = mock_vertex.submit_custom_job.call_args
        env_vars: dict[str, str] = submit_call.kwargs["env_vars"]
        assert "GCS_CODE_URI" in env_vars
        assert "GCS_DATA_URI" in env_vars
        assert "GCS_MODEL_URI" in env_vars
        assert env_vars["GCS_CODE_URI"].startswith("gs://")
