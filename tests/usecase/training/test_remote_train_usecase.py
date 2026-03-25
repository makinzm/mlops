"""
RemoteTrainUseCase の単体テスト。

なぜこのテストが必要か:
  - RemoteTrainUseCase は ObjectStorageRepository と TrainingJobRepository を受け取り、
    コード + データのオブジェクトストレージアップロード → リモート学習ジョブ送信 → 完了待機 →
    モデルのオブジェクトストレージダウンロード の一連の流れを担う。
  - 各リポジトリの Mock を使い、クラウド呼び出しなしで UseCase ロジックをテストする。
  - .gitignore 配置、RemoteTrainResult の正しい構造を保証する。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig, OmegaConf

from src.domain.repository.training_job import TrainingJobResult
from src.usecase.training.remote_train import RemoteTrainResult, RemoteTrainUseCase

_FAKE_COMMIT = "b" * 40
_FAKE_JOB_NAME = "projects/123/locations/asia-northeast1/customJobs/789"


def _make_cfg(tmp_path: Path) -> DictConfig:
    """RemoteTrainUseCase 用の DictConfig を生成する。"""
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
            "cloud": {
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
    mock.run_custom_job.return_value = TrainingJobResult(
        resource_name=_FAKE_JOB_NAME, state="SUCCEEDED"
    )
    mock.build_bootstrap_command.return_value = [
        "bash",
        "-c",
        "echo download && pip install -q deps && python /app/scripts/entrypoint.py",
    ]
    return mock


def _make_mock_git_repo() -> MagicMock:
    mock = MagicMock()
    mock.get_commit_hash.return_value = _FAKE_COMMIT
    return mock


class TestRemoteTrainUseCaseExecute:
    """RemoteTrainUseCase.execute() のテスト。"""

    def test_returns_remote_train_result(self, tmp_path: Path) -> None:
        """execute() が RemoteTrainResult を返すこと。"""
        cfg = _make_cfg(tmp_path)
        usecase = RemoteTrainUseCase(
            cfg=cfg,
            object_storage=_make_mock_gcs(),
            training_job=_make_mock_vertex(),
            git_repo=_make_mock_git_repo(),
        )
        result = usecase.execute()
        assert isinstance(result, RemoteTrainResult)

    def test_uploads_code_and_data_to_gcs(self, tmp_path: Path) -> None:
        """コードとデータがストレージにアップロードされること。"""
        cfg = _make_cfg(tmp_path)
        mock_gcs = _make_mock_gcs()
        usecase = RemoteTrainUseCase(
            cfg=cfg,
            object_storage=mock_gcs,
            training_job=_make_mock_vertex(),
            git_repo=_make_mock_git_repo(),
        )
        usecase.execute()

        # upload_dir が 4 回呼ばれていること（src, conf, scripts, data）
        assert mock_gcs.upload_dir.call_count == 4
        upload_uris = [call.args[1] for call in mock_gcs.upload_dir.call_args_list]
        assert any("/code/src" in uri for uri in upload_uris)
        assert any("/code/conf" in uri for uri in upload_uris)
        assert any("/code/scripts" in uri for uri in upload_uris)
        assert any("/data" in uri for uri in upload_uris)

    def test_submits_job_with_correct_params(self, tmp_path: Path) -> None:
        """リモート学習ジョブが正しいコンテナ URI とマシンタイプで送信されること。"""
        cfg = _make_cfg(tmp_path)
        mock_vertex = _make_mock_vertex()
        usecase = RemoteTrainUseCase(
            cfg=cfg,
            object_storage=_make_mock_gcs(),
            training_job=mock_vertex,
            git_repo=_make_mock_git_repo(),
        )
        usecase.execute()

        run_call = mock_vertex.run_custom_job.call_args
        assert run_call.kwargs["container_uri"] == "gcr.io/test/training:latest"
        assert run_call.kwargs["machine_type"] == "n1-standard-4"
        assert run_call.kwargs["service_account"] == "sa@test.iam.gserviceaccount.com"

    def test_run_custom_job_is_called(self, tmp_path: Path) -> None:
        """run_custom_job が呼ばれること。"""
        cfg = _make_cfg(tmp_path)
        mock_vertex = _make_mock_vertex()
        usecase = RemoteTrainUseCase(
            cfg=cfg,
            object_storage=_make_mock_gcs(),
            training_job=mock_vertex,
            git_repo=_make_mock_git_repo(),
        )
        usecase.execute()

        mock_vertex.run_custom_job.assert_called_once()

    def test_downloads_model_artifacts_from_gcs(self, tmp_path: Path) -> None:
        """モデル成果物がオブジェクトストレージからローカルにダウンロードされること。"""
        cfg = _make_cfg(tmp_path)
        mock_gcs = _make_mock_gcs()
        usecase = RemoteTrainUseCase(
            cfg=cfg,
            object_storage=mock_gcs,
            training_job=_make_mock_vertex(),
            git_repo=_make_mock_git_repo(),
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
        usecase = RemoteTrainUseCase(
            cfg=cfg,
            object_storage=_make_mock_gcs(),
            training_job=_make_mock_vertex(),
            git_repo=_make_mock_git_repo(),
        )
        usecase.execute()

        job_dir = Path(str(cfg.output_dir)) / str(cfg.job_id)
        assert (job_dir / ".gitignore").exists()

    def test_raises_on_job_failure(self, tmp_path: Path) -> None:
        """リモート学習ジョブが FAILED の場合に RuntimeError が送出されること。"""
        cfg = _make_cfg(tmp_path)
        mock_vertex = _make_mock_vertex()
        mock_vertex.run_custom_job.return_value = TrainingJobResult(
            resource_name=_FAKE_JOB_NAME, state="FAILED", error_message="OOM error"
        )
        usecase = RemoteTrainUseCase(
            cfg=cfg,
            object_storage=_make_mock_gcs(),
            training_job=mock_vertex,
            git_repo=_make_mock_git_repo(),
        )

        with pytest.raises(RuntimeError, match="OOM error"):
            usecase.execute()

    def test_result_contains_commit_hash(self, tmp_path: Path) -> None:
        """RemoteTrainResult に commit_hash が記録されること。"""
        cfg = _make_cfg(tmp_path)
        usecase = RemoteTrainUseCase(
            cfg=cfg,
            object_storage=_make_mock_gcs(),
            training_job=_make_mock_vertex(),
            git_repo=_make_mock_git_repo(),
        )
        result = usecase.execute()
        assert result.commit_hash == _FAKE_COMMIT

    def test_result_contains_gcs_uris(self, tmp_path: Path) -> None:
        """RemoteTrainResult にオブジェクトストレージ URI が記録されること。"""
        cfg = _make_cfg(tmp_path)
        usecase = RemoteTrainUseCase(
            cfg=cfg,
            object_storage=_make_mock_gcs(),
            training_job=_make_mock_vertex(),
            git_repo=_make_mock_git_repo(),
        )
        result = usecase.execute()
        assert result.gcs_data_uri.startswith("gs://")
        assert result.gcs_model_uri.startswith("gs://")

    def test_command_uses_bootstrap_from_infra(self, tmp_path: Path) -> None:
        """build_bootstrap_command の戻り値が command に渡されること。"""
        cfg = _make_cfg(tmp_path)
        mock_vertex = _make_mock_vertex()
        usecase = RemoteTrainUseCase(
            cfg=cfg,
            object_storage=_make_mock_gcs(),
            training_job=mock_vertex,
            git_repo=_make_mock_git_repo(),
        )
        usecase.execute()

        mock_vertex.build_bootstrap_command.assert_called_once()
        submit_call = mock_vertex.run_custom_job.call_args
        command: list[str] = submit_call.kwargs["command"]
        assert command == mock_vertex.build_bootstrap_command.return_value

    def test_env_vars_contain_gcs_uris(self, tmp_path: Path) -> None:
        """リモート学習ジョブの env_vars に GCS_DATA_URI と GCS_MODEL_URI が含まれること。"""
        cfg = _make_cfg(tmp_path)
        mock_vertex = _make_mock_vertex()
        usecase = RemoteTrainUseCase(
            cfg=cfg,
            object_storage=_make_mock_gcs(),
            training_job=mock_vertex,
            git_repo=_make_mock_git_repo(),
        )
        usecase.execute()

        submit_call = mock_vertex.run_custom_job.call_args
        env_vars: dict[str, str] = submit_call.kwargs["env_vars"]
        assert "GCS_DATA_URI" in env_vars
        assert "GCS_MODEL_URI" in env_vars
        assert "PYTHONPATH" in env_vars
