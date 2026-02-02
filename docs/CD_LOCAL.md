# ローカル環境でのCD実行

GitHub Actionsを使わずに、ローカル環境でCDパイプラインを実行する手順です。

---

## 前提条件

### 必須ツール
- [devbox](https://www.jetpack.io/devbox/) - 開発環境管理
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) - GCP操作
- Docker - イメージビルド

### 必須アカウント
- [Google Cloud Platform](https://cloud.google.com/)アカウント
- 課金が有効なGCPプロジェクト（またはこれから作成）
- [Kaggle](https://www.kaggle.com/)アカウント

### 認証情報
- GCPユーザー認証（`gcloud auth login`）
- Kaggle API認証情報（`~/.kaggle/kaggle.json`）

---

## 1. GCP初期セットアップ

### 1.0 gcloud CLIの初期認証

GCPの操作を行う前に、まずgcloud CLIにログインします。

```bash
# ブラウザが開き、Googleアカウントでログイン
gcloud auth login

# 認証状態の確認
gcloud auth list
```

**GCP Console確認:**
- ログイン成功後、[GCP Console](https://console.cloud.google.com/) にアクセスできることを確認

---

### 1.1 GCPプロジェクトIDの決定

プロジェクトIDはGCP全体でユニークである必要があります。以下のルールに従って決定してください。

**命名規則:**
- 6～30文字
- 小文字、数字、ハイフンのみ使用可能
- 先頭は小文字、末尾はハイフンは不可
- 例: `my-mlops-project-20260202`, `mlops-dev-123456`

**既存プロジェクトがある場合の確認方法:**
```bash
# 既存プロジェクト一覧を表示
gcloud projects list

# 現在選択されているプロジェクトIDを確認
gcloud config get-value project
```

### 1.2 GCPプロジェクト作成

```bash
# プロジェクトIDを環境変数に設定（以降のコマンドで使用）
export PROJECT_ID="your-unique-project-id"
```

```bash
# プロジェクト作成（初回のみ）
gcloud projects create $PROJECT_ID --name="MLOps Project"

# プロジェクト選択
gcloud config set project $PROJECT_ID
```

**課金アカウントの紐付け:**

```bash
# 課金アカウント一覧を表示
gcloud billing accounts list
# 出力例:
# ACCOUNT_ID            NAME                OPEN  MASTER_ACCOUNT_ID
# 01AB23-CD45EF-67890A  My Billing Account  True

# 課金アカウントIDを環境変数に設定（上記の ACCOUNT_ID をコピー）
export BILLING_ACCOUNT_ID="01AB23-CD45EF-67890A"

# プロジェクトに課金アカウントを紐付け
gcloud billing projects link $PROJECT_ID --billing-account=$BILLING_ACCOUNT_ID
```

**GCP Console確認:**
- [プロジェクト一覧](https://console.cloud.google.com/projectselector2)
- 作成したプロジェクトが表示され、「課金」列に課金アカウント名が表示されていることを確認
```

### 1.3 必要なAPIの有効化

```bash
gcloud services enable \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  storage.googleapis.com
```

**GCP Console確認:**
- [APIとサービス](https://console.cloud.google.com/apis/dashboard) → 「有効なAPIとサービス」
- 上記4つのAPIが有効化されていることを確認

### 1.4 Artifact Registryリポジトリ作成

```bash
gcloud artifacts repositories create mlops-training \
  --repository-format=docker \
  --location=asia-northeast1 \
  --description="MLOps training images"
```

**GCP Console確認:**
- [Artifact Registry](https://console.cloud.google.com/artifacts)
- `mlops-training` リポジトリが `asia-northeast1` に作成されていることを確認

### 1.5 GCSバケット作成

```bash
gsutil mb -l asia-northeast1 gs://$PROJECT_ID-mlops-dev
```

**GCP Console確認:**
- [Cloud Storage](https://console.cloud.google.com/storage/browser)
- `{プロジェクトID}-mlops-dev` バケットが作成されていることを確認

### 1.6 サービスアカウント作成（CI/CD用）

```bash
# サービスアカウント作成
gcloud iam service-accounts create mlops-cicd \
  --display-name="MLOps CI/CD Service Account"

# 必要な権限を付与
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

**GCP Console確認:**
- [IAMと管理](https://console.cloud.google.com/iam-admin/serviceaccounts) → 「サービスアカウント」
- `mlops-cicd` サービスアカウントが作成されていることを確認
- [IAM](https://console.cloud.google.com/iam-admin/iam) → 上記4つのロールが付与されていることを確認

### 1.7 ローカル認証設定

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
