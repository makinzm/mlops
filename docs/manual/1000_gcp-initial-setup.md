# GCP 初期セットアップガイド（完全初心者向け）

このガイドは、GCP（Google Cloud Platform）を一度も使ったことがない方向けに、
Vertex AI でモデルを学習するための環境を一から作る手順を解説します。

---

## 1. GCP とは何か

GCP（Google Cloud Platform）は Google が提供するクラウドコンピューティングサービスです。
自分のパソコンの代わりに、Google のデータセンターのサーバーを借りて計算を実行できます。
このプロジェクトでは、LightGBM の学習を GCP の Vertex AI というサービスで実行します。
Vertex AI を使うと、ローカル PC よりも高性能なマシンで学習でき、
学習中もパソコンをシャットダウンできます。

---

## 2. Google アカウントの作成・GCP コンソールへのアクセス

1. https://accounts.google.com でアカウントを作成（既存の Gmail アカウントでも可）
2. https://console.cloud.google.com にアクセス
3. Google アカウントでサインイン

---

## 3. プロジェクトの作成

GCP ではすべてのリソースが「プロジェクト」単位で管理されます。

1. コンソール上部のプロジェクト選択ドロップダウン（「My Project」などと表示されている部分）をクリック
2. 「新しいプロジェクト」をクリック
3. プロジェクト名を入力（例: `mlops-titanic`）
4. 「作成」をクリック
5. 作成後、プロジェクト ID を `terraform/terraform.tfvars` の `project_id` にメモする（例: `mlops-titanic-123456`）

> **機密性: 低リスク（ただし Git にはコミットしない）**
>
> プロジェクト ID を知られただけではリソースにアクセスできません（IAM で保護）。
> ただし Google はプロジェクト ID を **PII（個人識別情報）** に分類しており、
> 公開すると偵察（バケット名の推測等）の起点にされるリスクがあります。
> `terraform.tfvars` は `.gitignore` 済みのため Git にコミットされません。
>
> 参考:
> - [AIP-2510: Project identifiers](https://google.aip.dev/cloud/2510) — Google 内部 API 設計標準。プロジェクト ID を PII と明記。
> - [Creating and managing projects | Resource Manager](https://cloud.google.com/resource-manager/docs/creating-managing-projects) — 「プロジェクト名や ID に PII やセキュリティデータを含めないこと」と記載。

---

## 4. 請求アカウントの有効化

GCP の多くのサービスは有料です。ただし、新規アカウントには **$300 の無料クレジット**（90日間有効）が付与されます。
Vertex AI の学習は無料枠の範囲内で十分試せます。

1. コンソール左上のハンバーガーメニュー → 「お支払い」
2. 「請求先アカウントをリンク」または「請求先アカウントを作成」をクリック
3. クレジットカード情報を入力（無料枠内は請求されません）
4. 請求アカウント ID を `terraform/terraform.tfvars` の `billing_account_id` にメモする（例: `012345-ABCDEF-789012`）

> **機密性: 低リスク（ただし Git にはコミットしない）**
>
> 請求アカウント ID を知られただけでは課金操作はできません（IAM ロール `Billing Account Administrator` 等が必要）。
> 公開する理由もないため `terraform.tfvars`（`.gitignore` 済み）で管理します。
>
> 参考:
> - [Cloud Billing access control](https://cloud.google.com/billing/docs/how-to/billing-access) — 請求操作は IAM で保護されており、ID の秘匿ではなくロールで制御される。
> - [Find a Cloud Billing account ID](https://cloud.google.com/billing/docs/how-to/find-billing-account-id) — ID はシステム生成の識別子。`billing.accounts.get` 権限を持つユーザーのみ閲覧可能。

---

## 5. 必要な API の有効化

以下の API を有効にします。**コンソールから有効化する方法**と **gcloud コマンド**の両方を記載します。

### コンソールから有効化する場合

コンソール左上のハンバーガーメニュー → 「APIとサービス」→「ライブラリ」で各 API を検索して「有効にする」をクリック。

有効化が必要な API:
- Vertex AI API
- Cloud Storage API
- Artifact Registry API
- Cloud Functions API
- Pub/Sub API
- Cloud Billing API

### gcloud コマンドで一括有効化する場合

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  cloudfunctions.googleapis.com \
  pubsub.googleapis.com \
  billingbudgets.googleapis.com \
  --project YOUR_PROJECT_ID
```

---

## 6. gcloud CLI のインストール

### Linux（推奨）

```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

### macOS

```bash
brew install --cask google-cloud-sdk
```

### Windows

https://cloud.google.com/sdk/docs/install の指示に従いインストーラーをダウンロード。

---

## 7. gcloud の認証設定

### ユーザー認証

```bash
gcloud auth login
```

ブラウザが開くので Google アカウントでサインイン。

### アプリケーション認証（コード実行用）

Python SDK（google-cloud-storage 等）が使う認証情報を設定します。

```bash
gcloud auth application-default login
```

### プロジェクトの設定

```bash
gcloud config set project YOUR_PROJECT_ID
```

---

## 8. Terraform のインストール

Terraform は Infrastructure as Code ツールです。GCS バケットや SA などを自動作成します。

### tfenv（バージョン管理ツール）を使う方法（推奨）

```bash
# tfenv のインストール
git clone --depth=1 https://github.com/tfutils/tfenv.git ~/.tfenv
echo 'export PATH="$HOME/.tfenv/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Terraform 1.6 のインストール
tfenv install 1.6.0
tfenv use 1.6.0

# バージョン確認
terraform --version
```

### 直接インストールする場合

https://developer.hashicorp.com/terraform/install からプラットフォームに合ったバイナリをダウンロード。

---

## 9. Docker のインストール

学習コンテナをビルド・プッシュするために必要です。

### Linux

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# ログアウトして再ログイン後に有効
```

### macOS

https://www.docker.com/products/docker-desktop から Docker Desktop をダウンロード。

---

## 10. .env ファイルへの GCP 設定の追加

`.env.example` をコピーして `.env` を作成し、GCP の設定を追記します。

```bash
cp .env.example .env
```

`.env` に追記:

```dotenv
# GCP 設定
GCP_PROJECT=your-project-id
GCP_REGION=asia-northeast1
```

`.env` は `.gitignore` 済みのため Git にコミットされません。

---

## 次のステップ

GCP の初期設定が完了したら、次は Terraform でリソースを作成します。
→ [docs/manual/1001_vertex-ai-training.md](./1001_vertex-ai-training.md)
