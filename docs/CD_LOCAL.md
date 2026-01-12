# ローカル環境でのCD実行

GitHub Actionsを使わずに、ローカル環境でCDパイプラインを実行する手順です。

---

## 前提条件

### 必須ツール
- [devbox](https://www.jetpack.io/devbox/) - 開発環境管理
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) - GCP操作
- Docker - イメージビルド

### 認証情報
- Kaggle API認証情報（`~/.kaggle/kaggle.json`）
- GCPサービスアカウント認証

---

## 1. GCP初期セットアップ

### 1.1 GCPプロジェクト作成

```bash
# プロジェクト作成（初回のみ）
gcloud projects create YOUR_PROJECT_ID --name="MLOps Project"

# プロジェクト選択
gcloud config set project YOUR_PROJECT_ID

# 課金アカウントの紐付け（コンソールで実施するか以下コマンド）
gcloud billing accounts list
gcloud billing projects link YOUR_PROJECT_ID --billing-account=BILLING_ACCOUNT_ID
```

### 1.2 必要なAPIの有効化

```bash
gcloud services enable \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  storage.googleapis.com
```

### 1.3 Artifact Registryリポジトリ作成

```bash
gcloud artifacts repositories create mlops-training \
  --repository-format=docker \
  --location=asia-northeast1 \
  --description="MLOps training images"
```

### 1.4 GCSバケット作成

```bash
gsutil mb -l asia-northeast1 gs://YOUR_PROJECT_ID-mlops-dev
```

### 1.5 サービスアカウント作成（CI/CD用）

```bash
# サービスアカウント作成
gcloud iam service-accounts create mlops-cicd \
  --display-name="MLOps CI/CD Service Account"

# 必要な権限を付与
PROJECT_ID=$(gcloud config get-value project)
SA_EMAIL="mlops-cicd@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/cloudbuild.builds.builder"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/storage.admin"
```

### 1.6 ローカル認証設定

**方法A: Application Default Credentials（開発用・推奨）**
```bash
gcloud auth application-default login
```

**方法B: サービスアカウントキー（CI/CD再現用）**
```bash
# キー作成（セキュリティ上、本番では非推奨）
gcloud iam service-accounts keys create ~/gcp-key.json \
  --iam-account=$SA_EMAIL

# 環境変数設定
export GOOGLE_APPLICATION_CREDENTIALS=~/gcp-key.json
```

---

## 2. Kaggle初期セットアップ

### 2.1 API認証情報の取得

1. [Kaggle Account](https://www.kaggle.com/account) にアクセス
2. 「API」セクションで「Create New API Token」をクリック
3. ダウンロードされた`kaggle.json`を `~/.kaggle/` に配置

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2.2 環境変数設定

```bash
export KAGGLE_COMPETITION="your-competition-name"
export KAGGLE_USERNAME="your-kaggle-username"
```

---

## 3. 環境変数の設定

ローカル実行に必要な環境変数をまとめて設定します。

```bash
# GCP設定
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="asia-northeast1"

# Kaggle設定
export KAGGLE_COMPETITION="your-competition-name"
export KAGGLE_USERNAME="your-kaggle-username"

# オプション: .envファイルに保存
cat <<EOF > .env.local
GCP_PROJECT_ID=your-project-id
GCP_REGION=asia-northeast1
KAGGLE_COMPETITION=your-competition-name
KAGGLE_USERNAME=your-kaggle-username
EOF

# 読み込み
source .env.local
```

---

## 4. ローカルCDパイプライン実行

### 4.1 依存関係インストール

```bash
make sync EXTRA=kaggle
```

### 4.2 Step 1: Kaggle Dataset Push

```bash
make kaggle-push-dataset MSG="Local test: $(date +%Y%m%d-%H%M%S)"
```

### 4.3 Step 2: Docker イメージビルド & Push

```bash
# ローカルビルドのみ
make docker-build TAG=local-$(date +%Y%m%d)

# Cloud Build経由でビルド & Artifact Registryにpush
make gcloud-build TAG=local-$(date +%Y%m%d)
```

### 4.4 Step 3: Vertex AI訓練ジョブ投入

```bash
make gcp-train TAG=local-$(date +%Y%m%d)
```

訓練ジョブの状態確認:
```bash
gcloud ai custom-jobs list --region=$GCP_REGION
```

### 4.5 Step 4: モデルダウンロード

```bash
make gcp-download-model MODEL=latest
```

### 4.6 Step 5: 推論 & Kaggle提出

```bash
make kaggle-inference-submit MSG="Local submission: $(date +%Y%m%d-%H%M%S)"
```

---

## 5. ワンライナー実行

全ステップを一括実行する場合：

```bash
# 全パイプライン実行（訓練完了を待たない簡易版）
TAG="local-$(date +%Y%m%d%H%M%S)" && \
make kaggle-push-dataset MSG="Commit: $TAG" && \
make gcloud-build TAG=$TAG && \
make gcp-train TAG=$TAG && \
echo "Training job submitted. Wait for completion, then run:" && \
echo "make gcp-download-model MODEL=latest && make kaggle-inference-submit MSG=$TAG"
```

---

## 6. トラブルシューティング

### GCP認証エラー

```
ERROR: (gcloud.ai.custom-jobs.create) PERMISSION_DENIED
```
→ `gcloud auth application-default login` を再実行

### Kaggle API エラー

```
OSError: Could not find kaggle.json
```
→ `~/.kaggle/kaggle.json` が存在し、権限が600であることを確認

### Cloud Build タイムアウト

```bash
# タイムアウトを延長してビルド
gcloud builds submit \
  --timeout=1800s \
  --tag $GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/mlops-training/train:$TAG \
  --file docker/Dockerfile.train .
```

### Vertex AI ジョブ確認

```bash
# ジョブ一覧
gcloud ai custom-jobs list --region=$GCP_REGION

# 特定ジョブの詳細
gcloud ai custom-jobs describe JOB_ID --region=$GCP_REGION

# ログ確認
gcloud ai custom-jobs stream-logs JOB_ID --region=$GCP_REGION
```
