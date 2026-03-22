# GCP コスト監視ガイド

Vertex AI は実行時間に応じてコストが発生します。
このガイドでは、コストの確認方法・予算アラートの設定・緊急時の対処法を説明します。

---

## 1. コスト確認方法（GCP コンソール）

### 請求の確認

1. GCP コンソール → 左上メニュー → 「お支払い」
2. 「コスト管理」→「コストレポート」
3. プロジェクトを選択してコストを確認

### リアルタイムコスト確認

1. コンソール → 「お支払い」
2. 「概要」の「今月の費用」にリアルタイムの費用が表示されます

### Vertex AI 固有のコスト確認

1. コンソール → 「Vertex AI」
2. 「請求」ページまたは「使用状況」でサービス別の使用量を確認

---

## 2. 予算アラートメールの確認

Terraform で設定した `budget_alert_email` に対して、予算の 50% / 80% / 100% / 120% 到達時に
メールが送信されます。

メールの件名例:
```
Budget alert: MLOps Monthly Budget - You've reached 80% of your budget
```

---

## 3. budget_action の切り替え方法

`conf/gcp/vertex.yaml` の `budget_action` を変更することで動作を切り替えられます。

```yaml
gcp:
  budget_action: warn   # "warn": 通知のみ（デフォルト）
  # budget_action: stop  # "stop": 予算超過時にジョブを自動停止
```

変更後は Terraform を再適用:

```bash
cd terraform
terraform apply
```

---

## 4. Vertex AI ジョブの手動停止（緊急時）

### GCP コンソールから停止

1. GCP コンソール → 「Vertex AI」→「トレーニング」→「カスタムジョブ」
2. 停止したいジョブを選択
3. 「キャンセル」ボタンをクリック

### gcloud コマンドから停止

実行中のジョブ一覧を確認:
```bash
gcloud ai custom-jobs list --project=YOUR_PROJECT_ID --region=asia-northeast1 \
  --filter="state=JOB_STATE_RUNNING"
```

特定のジョブを停止:
```bash
gcloud ai custom-jobs cancel JOB_ID \
  --project=YOUR_PROJECT_ID \
  --region=asia-northeast1
```

---

## 5. 使い終わったリソースの削除

### 全リソースの削除（terraform destroy）

```bash
cd terraform
terraform destroy
```

この操作で以下が削除されます:
- GCS バケット（バケット内のファイルも削除）
- Artifact Registry リポジトリ（コンテナイメージも削除）
- サービスアカウント
- Pub/Sub トピック
- Cloud Functions
- 予算アラート

**注意**: `force_destroy = false` に設定されているため、バケットに中身がある場合は削除が失敗します。
先にバケットの中身を削除してから実行してください:

```bash
gsutil -m rm -r gs://your-bucket-name/**
terraform destroy
```

### Docker イメージのみ削除

```bash
gcloud artifacts docker images delete \
  asia-northeast1-docker.pkg.dev/your-project/mlops/training:latest
```

---

## 6. コスト削減のヒント

### マシンタイプの選択

`conf/gcp/vertex.yaml` の `machine_type` を変更してコストを削減できます:

| マシンタイプ | vCPU | RAM | 目安コスト/時間 |
|---|---|---|---|
| n1-standard-1 | 1 | 3.75 GB | 約 $0.05 |
| n1-standard-4 | 4 | 15 GB | 約 $0.19（デフォルト） |
| n1-standard-8 | 8 | 30 GB | 約 $0.38 |

小規模データセットや実験時は `n1-standard-1` や `n1-standard-4` で十分です。

### 不要なジョブの確認

以下のコマンドで実行中・キューイング中のジョブを確認できます:

```bash
gcloud ai custom-jobs list \
  --project=YOUR_PROJECT_ID \
  --region=asia-northeast1 \
  --filter="state=JOB_STATE_RUNNING OR state=JOB_STATE_QUEUED"
```

### early_stopping_rounds の活用

`conf/competition/titanic/training/lgbm.yaml` の `lgbm.early_stopping_rounds` を
小さくすることで、過学習が起きた時点で学習を早期終了してコストを削減できます。
