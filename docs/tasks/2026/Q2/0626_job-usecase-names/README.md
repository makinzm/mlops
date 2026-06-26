# TASK-008: usecase 層から `cloud` を除去し `job_*` に統一

## 背景

TASK-007 で `remote/vertex` → `cloud_*` にリネームしたが、`cloud` もインフラ用語であり
usecase 層には不適切という指摘を受けた。usecase 層の名前は「何をするか」だけを表すべき。

## 変更方針

`cloud_*` → `job_*` にリネーム。ドメイン概念としての「ジョブ」（スケジューラに投入できる
作業単位）は、実行場所（cloud/local/remote）に依存しない。

## リネーム対象

### usecase ファイル
- `src/usecase/training/cloud_train.py` → `job_train.py` (`CloudTrainUseCase` → `JobTrainUseCase`, `CloudTrainResult` → `JobTrainResult`)
- `src/usecase/training/cloud_submit.py` → `job_submit.py` (`CloudSubmitUseCase` → `JobSubmitUseCase`, `CloudSubmitResult` → `JobSubmitResult`)
- `src/usecase/training/cloud_download.py` → `job_download.py` (`CloudDownloadUseCase` → `JobDownloadUseCase`, `CloudDownloadResult` → `JobDownloadResult`)

### presentation 層
- `src/presentation/runners.py`: `run_cloud_train` → `run_job_train`, `run_cloud_submit` → `run_job_submit`, `run_cloud_download` → `run_job_download`
- `src/presentation/registry.py`: キー `"cloud_train"` / `"cloud_submit"` / `"cloud_download"` → `"job_train"` / `"job_submit"` / `"job_download"`
- `src/presentation/cloud_config.py`: `cloud_download_and_push` コメント等更新

### conf
- `conf/usecase/cloud_train.yaml` → `job_train.yaml`
- `conf/usecase/cloud_submit.yaml` → `job_submit.yaml`
- `conf/usecase/cloud_download.yaml` → `job_download.yaml`
- `conf/usecase/create_cloud_models.yaml` → `create_job_models.yaml`
- `conf/usecase/upload_cloud_models.yaml` → `upload_job_models.yaml`
- `conf/usecase/push_cloud_notebook.yaml` → `push_job_notebook.yaml`
- `conf/competition/titanic/pipeline/cloud_fire_and_forget.yaml` → `job_fire_and_forget.yaml`
- `conf/competition/titanic/pipeline/cloud_download_and_push.yaml` → `job_download_and_push.yaml`
- `conf/competition/titanic/pipeline/cloud_to_kaggle.yaml` → `job_to_kaggle.yaml`
- `conf/config.yaml`: `cloud_jobs_history_dir` → `job_history_dir`, コメント更新

### domain
- `src/domain/data/job_manifest.py`: `cloud_job_name` → `job_name` にリネーム（cloud も除去対象）
- `src/infrastructure/executor/factory.py`: エラーメッセージのコメント更新

### conf/infra
- `conf/cloud/vertex.yaml` → `conf/infra/vertex.yaml` に移動（旧ファイルは `git rm` で削除）
- `conf/config.yaml`: `cloud: null` → `infra: null`
- `conf/usecase/job_*.yaml`: `defaults: - /infra: vertex` に更新

### mille.toml
- `src_usecase.name_deny` に `"cloud"` を追加

### テスト
- `tests/usecase/training/test_cloud_train_usecase.py` → `test_job_train_usecase.py`
- `tests/usecase/training/test_cloud_submit_usecase.py` → `test_job_submit_usecase.py`
- `tests/usecase/training/test_cloud_download_usecase.py` → `test_job_download_usecase.py`
- `tests/presentation/test_registry.py`: キー更新
- `tests/usecase/pipeline/test_pipeline_remote.py`: `usecase: cloud_train` → `usecase: job_train`
- `tests/test_main_resolve.py`: recipe/dir 名更新

## TDD サイクル

1. RED: テストのクラス名・インポートを `job_*` に更新してコミット（`--no-verify`）
2. GREEN: 実装ファイルをリネーム・クラス名更新 → テスト通過 → コミット
3. REFACTOR: mille / ruff / ty clean 確認
