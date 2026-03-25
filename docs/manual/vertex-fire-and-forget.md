# Vertex AI Fire-and-Forget Training

## Overview

ジョブ投入後即座にローカルプロセスを終了し、完了時に Slack/Email 通知を受け、手動でモデルをダウンロードして推論・提出まで行う。

## Prerequisites

- GCP プロジェクトの設定が完了していること（`docs/manual/1000_gcp-initial-setup.md` 参照）
- `.env` に GCP 設定が記載されていること
- 前処理済みデータが存在すること

## Configuration

### 1. .env に GCP 設定を追加（未設定の場合）

```bash
# .env
GCP_PROJECT=mlops-titanic-123456
GCP_REGION=asia-northeast1
```

### 2. Slack 通知を有効にする（オプション）

Slack Incoming Webhook を作成し、`.env` に追加する:

```bash
# .env
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../xxx
```

### 3. Email 通知を有効にする（オプション）

```bash
# .env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_SENDER=noreply@example.com
SMTP_RECIPIENT=user@example.com
SMTP_USERNAME=noreply@example.com
SMTP_PASSWORD=your-app-password
```

## Usage

### Step 1: ジョブ投入（即座に終了）

```bash
# 単体実行
uv run python -m src usecase=vertex_submit recipe=lgbm

# パイプライン実行（前処理 -> 投入）
uv run python -m src usecase=pipeline recipe=vertex_fire_and_forget
```

実行後、以下が出力される:
- `models/{competition}/{job_id}/job_manifest.yaml` — ジョブ情報
- `models/{competition}/{job_id}/.gitignore` — データ除外設定

### Step 2: 通知を待つ

Slack/Email 通知が届くまで待機する。通知には以下が含まれる:
- Job ID
- Competition
- Recipe
- Commit hash
- ダウンロードコマンド例

### Step 3: モデルダウンロード

```bash
# manifest_path を指定してダウンロード
uv run python -m src usecase=vertex_download \
  manifest_path=models/titanic/titanic_lgbm/job_manifest.yaml
```

### Step 4: 推論 -> Kaggle 提出（パイプライン）

```bash
uv run python -m src usecase=pipeline recipe=vertex_download_and_push \
  manifest_path=models/titanic/titanic_lgbm/job_manifest.yaml
```

このパイプラインは以下を順に実行する:
1. `vertex_download` — GCS からモデルダウンロード
2. `inference` — submission.csv 生成
3. `update_source_dataset` — コードを Kaggle Dataset に push
4. `push_notebook` — 推論 Notebook を push

## Config Files

| File | Description |
|------|-------------|
| `conf/usecase/vertex_submit.yaml` | vertex_submit usecase 設定 |
| `conf/usecase/vertex_download.yaml` | vertex_download usecase 設定 |
| `conf/notification/slack.yaml` | Slack 通知設定 |
| `conf/notification/email.yaml` | Email 通知設定 |
| `conf/competition/titanic/pipeline/vertex_fire_and_forget.yaml` | 前処理 -> 投入パイプライン |
| `conf/competition/titanic/pipeline/vertex_download_and_push.yaml` | DL -> 推論 -> 提出パイプライン |

## Job Manifest

`job_manifest.yaml` のステータス遷移:

```
SUBMITTED -> SUCCEEDED -> DOWNLOADED
                       -> FAILED
```

manifest の内容例:

```yaml
job_id: titanic_lgbm
competition: titanic
recipe: lgbm
timestamp: "20260325T143000"
commit_hash: "abc123..."
status: SUBMITTED
remote_job_name: "projects/.../customJobs/789"
gcs_code_uri: "gs://bucket/jobs/.../code"
gcs_data_uri: "gs://bucket/jobs/.../data"
gcs_model_uri: "gs://bucket/jobs/.../models"
submitted_at: "2026-03-25T14:30:00"
```

## Troubleshooting

### ジョブがまだ実行中

```
RuntimeError: Remote training job is still RUNNING
```

→ しばらく待ってから再度 `vertex_download` を実行する。

### ジョブが失敗

```
RuntimeError: Remote training job FAILED
```

→ GCP Console でジョブログを確認する:
```bash
gcloud ai custom-jobs describe {job_name} --project={project} --region={region}
```
