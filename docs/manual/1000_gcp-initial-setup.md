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
5. 作成後、プロジェクト ID をメモしておく（例: `mlops-titanic-123456`）
   → **メモ先: `.env` の `GCP_PROJECT`**（セクション 10 で作成）

> **機密性: 低リスク（ただし Git にはコミットしない）**
>
> プロジェクト ID を知られただけではリソースにアクセスできません（IAM で保護）。
> ただし Google はプロジェクト ID を **PII（個人識別情報）** に分類しており、
> 公開すると偵察（バケット名の推測等）の起点にされるリスクがあります。
> `.env` は `.gitignore` 済みのため Git にコミットされません。
>
> 参考:
> - [AIP-2510: Project identifiers](https://google.aip.dev/cloud/2510) — Google 内部 API 設計標準。プロジェクト ID を PII と明記。
> - [Creating and managing projects | Resource Manager](https://cloud.google.com/resource-manager/docs/creating-managing-projects) — 「プロジェクト名や ID に PII やセキュリティデータを含めないこと」と記載。

---

## 4. 請求アカウントの有効化

GCP の多くのサービスは有料です。ただし、新規アカウントには **$300 の無料クレジット**（90日間有効）が付与されます。
Vertex AI の学習は無料枠の範囲内で十分試せます。

1. コンソール左上のハンバーガーメニュー → 「請求」→「請求アカウントを管理」
2. 「請求先アカウントをリンク」または「請求先アカウントを作成」をクリック
3. クレジットカード情報を入力（無料枠内は請求されません）
4. 請求アカウントがプロジェクトにリンクされていることを確認

予算アラートの設定方法は [1002_cost-monitoring.md](./1002_cost-monitoring.md) を参照してください。

---

## 5. gcloud CLI のインストール

[クイックスタート: Google Cloud CLI をインストールする  |  Google Cloud SDK  |  Google Cloud Documentation](https://docs.cloud.google.com/sdk/docs/install-sdk?hl=ja#deb)

## 6. gcloud の認証設定

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

## 7. 必要な API の有効化

以下の API を有効にします。**コンソールから有効化する方法**と **gcloud コマンド**の両方を記載します。

### コンソールから有効化する場合

コンソール左上のハンバーガーメニュー → 「APIとサービス」→「ライブラリ」で各 API を検索して「有効にする」をクリック。

有効化が必要な API:

| API | 用途 |
|-----|------|
| Vertex AI API | モデル学習ジョブ |
| Cloud Storage API | データ・モデルのステージング |
| Artifact Registry API | Docker イメージ保管 |
| IAM API | サービスアカウント作成 |
| Cloud Resource Manager API | IAM ポリシーの読み書き |
| Compute Engine API | Vertex AI のマシン割り当て |

### gcloud コマンドで一括有効化する場合

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  cloudresourcemanager.googleapis.com \
  compute.googleapis.com
```

> **重要**: 全ての API を有効化してから `terraform apply` に進んでください。
> API が1つでも漏れると `terraform apply` でエラーになります。

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

`.env` に追記（セクション 3 でメモしたプロジェクト ID をここに記入する）:

```dotenv
# GCP 設定（セクション 3 でメモした値）
GCP_PROJECT=your-project-id                # 【必須】セクション 3 で取得したプロジェクト ID
GCP_REGION=asia-northeast1                 # リージョン（下記の選び方を参照）
# GCP_BUCKET_NAME=my-custom-bucket-name    # 未設定なら {GCP_PROJECT}-mlops-staging
```

`.env` は `.gitignore` 済みのため Git にコミットされません。
次の手順（[1001_vertex-ai-training.md](./1001_vertex-ai-training.md)）でスクリプトが `.env` から `terraform.tfvars` を自動生成します。

### GCP_REGION の選び方

リージョンは「どこのデータセンターを使うか」です。以下の基準で選んでください:

| 基準 | 説明 |
|------|------|
| **レイテンシ** | 自分に近いリージョンほどデータ転送が速い |
| **料金** | リージョンによって同じマシンでも料金が異なる |
| **Vertex AI の対応** | 全リージョンで Vertex AI が使えるわけではない |

**日本から使う場合の推奨:**

| リージョン | 場所 | 特徴 |
|-----------|------|------|
| `asia-northeast1` | 東京 | 最も低レイテンシ。**迷ったらこれ** |
| `us-central1` | アイオワ | Vertex AI の全機能が最初に利用可能になる。料金も安め |

Vertex AI の対応リージョン一覧:
[Vertex AI locations](https://cloud.google.com/vertex-ai/docs/general/locations)

---

## 次のステップ

GCP の初期設定が完了したら、次は Terraform でリソースを作成します。
→ [docs/manual/1001_vertex-ai-training.md](./1001_vertex-ai-training.md)
