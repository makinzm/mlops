# Vertex AI Fire-and-Forget Training

## Context

現在の `usecase=remote_train` (RemoteTrainUseCase) は Vertex AI ジョブの完了までローカルでブロックする。
学習に数時間かかる場合、ローカルプロセスを開きっぱなしにする必要がある。

## Goal

ジョブ投入後即座にローカルプロセスを終了し、完了時に通知を受け、手動でモデルをダウンロードして推論・提出まで行えるようにする。

## Flow

```
1. vertex_submit  -> GCS upload -> CustomJob submit(sync=False) -> job_manifest.yaml save -> exit
2. [Vertex container] -> training -> model GCS upload -> Slack/Email notification
3. User receives notification
4. vertex_download -> manifest read -> status check -> GCS model download -> manifest update
5. pipeline recipe=vertex_download_and_push -> inference -> update_dataset -> push_notebook
```

## New Files

### Domain Layer
- `src/domain/repository/notifier.py` - Notifier Protocol + NotificationPayload dataclass
- `src/domain/data/job_manifest.py` - JobManifest dataclass

### Infrastructure Layer
- `src/infrastructure/notifier/__init__.py` - Package
- `src/infrastructure/notifier/slack_notifier.py` - SlackNotifier (urllib.request HTTP POST)
- `src/infrastructure/notifier/email_notifier.py` - EmailNotifier (smtplib)
- `src/infrastructure/notifier/composite_notifier.py` - CompositeNotifier

### UseCase Layer
- `src/usecase/training/vertex_submit.py` - VertexSubmitUseCase
- `src/usecase/training/vertex_download.py` - VertexDownloadUseCase

### Config
- `conf/usecase/vertex_submit.yaml`
- `conf/usecase/vertex_download.yaml`
- `conf/notification/slack.yaml`
- `conf/notification/email.yaml`
- `conf/competition/titanic/pipeline/vertex_fire_and_forget.yaml`
- `conf/competition/titanic/pipeline/vertex_download_and_push.yaml`

### Tests
- `tests/domain/repository/test_notifier.py`
- `tests/infrastructure/notifier/test_slack_notifier.py`
- `tests/infrastructure/notifier/test_email_notifier.py`
- `tests/infrastructure/notifier/test_composite_notifier.py`
- `tests/usecase/training/test_vertex_submit_usecase.py`
- `tests/usecase/training/test_vertex_download_usecase.py`

## Existing File Changes
- `src/domain/repository/training_job.py` - submit_custom_job() + get_job_status() methods
- `src/infrastructure/gcp/vertex_ai.py` - implementation of above 2 methods
- `scripts/remote_entrypoint.py` - Slack/Email notification after training
- `src/main.py` - vertex_submit, vertex_download dispatch + pipeline runner registration
- `conf/config.yaml` - notification: null, manifest_path: null keys
- `.env.example` - SLACK_WEBHOOK_URL, SMTP env vars

## Implementation Order (TDD)
1. RED: All tests
2. GREEN: Domain -> Infrastructure -> UseCase -> main.py -> config -> entrypoint
3. Manual docs
