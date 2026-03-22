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

---

## 2. 予算アラートの設定（コンソールから手動）

> **なぜ Terraform で管理しないのか？**
>
> Billing Budget の Terraform 作成には billing account レベルの
> `roles/billing.budgets.admin` 権限が必要ですが、個人プロジェクトでは
> この権限の付与が困難です（API が 400 を返す既知の問題）。
> そのため、予算アラートは GCP コンソールから手動で設定します。
>
> 参考: [hashicorp/terraform-provider-google #9375](https://github.com/hashicorp/terraform-provider-google/issues/9375)

### 手順

1. GCP コンソール → 「お支払い」→「予算とアラート」
2. 「予算を作成」をクリック
3. 以下を設定:
   - **名前**: `MLOps Monthly Budget`
   - **プロジェクト**: 自分のプロジェクトを選択
   - **金額**: 月間上限（例: $10）
   - **アラートのしきい値**: 50%, 80%, 100%
4. 「保存」をクリック

設定したメールアドレスに、しきい値到達時にアラートメールが届きます。

---

## 3. Vertex AI ジョブの手動停止（緊急時）

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

## 4. 使い終わったリソースの削除

### 全リソースの削除（terraform destroy）

```bash
cd terraform
terraform destroy
```

この操作で以下が削除されます:
- GCS バケット
- Artifact Registry リポジトリ（コンテナイメージも削除）
- サービスアカウント

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

## 5. コスト削減のヒント

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
