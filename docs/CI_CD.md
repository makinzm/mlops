# CI/CD パイプライン

本プロジェクトのGitHub Actionsワークフローの説明です。

## ワークフロー一覧

| ワークフロー | トリガー | 概要 |
|-------------|---------|------|
| CI | PR/Push to main | コード品質チェック（lint, test, typecheck） |
| CD - Deploy and Train | Push to main / 手動 | Kaggle push + Vertex AI訓練 + 自動提出 |
| Manual Kaggle Submit | 手動 | 任意のモデルでKaggle提出 |

---

## CI (ci.yaml)

### トリガー
- `main`ブランチへのPull Request
- `main`ブランチへのPush

### 実行内容

```
┌─────────────────────────────────────────────────┐
│  lint-test job                                  │
├─────────────────────────────────────────────────┤
│  1. make sync EXTRA=dev     # 依存関係インストール │
│  2. make lint               # ruff check        │
│  3. make typecheck          # ty check          │
│  4. make cov-xml            # pytest + coverage │
│  5. Upload coverage to Codecov (push時のみ)     │
└─────────────────────────────────────────────────┘
```

### 使用するMakeターゲット
- `make sync EXTRA=dev` - 開発依存関係のインストール
- `make lint` - コードスタイルチェック
- `make typecheck` - 型チェック
- `make cov-xml` - テスト実行とカバレッジ出力

---

## CD - Deploy and Train (cd-main.yaml)

### トリガー
- `main`ブランチへのPush（自動）
- 手動実行（`workflow_dispatch`）
  - `skip_training`: Vertex AI訓練をスキップするオプション

### 実行内容

```
┌─────────────────────┐     ┌─────────────────────┐
│  kaggle-push job    │     │  vertex-train job   │
│  (並列実行)         │     │  (並列実行)         │
├─────────────────────┤     ├─────────────────────┤
│ 1. make sync        │     │ 1. GCP認証          │
│    EXTRA=kaggle     │     │ 2. make gcloud-build│
│ 2. make kaggle-setup│     │    TAG=<sha>        │
│ 3. make kaggle-push │     │ 3. make gcp-train   │
│    -dataset         │     │    TAG=<sha>        │
└─────────┬───────────┘     └──────────┬──────────┘
          │                            │
          └──────────┬─────────────────┘
                     ▼
          ┌─────────────────────┐
          │  kaggle-submit job  │
          │  (両方完了後)       │
          ├─────────────────────┤
          │ 1. make sync        │
          │    EXTRA=kaggle     │
          │ 2. make kaggle-setup│
          │ 3. GCP認証          │
          │ 4. make gcp-download│
          │    -model MODEL=    │
          │    latest           │
          │ 5. make kaggle-     │
          │    inference-submit │
          │    MSG=auto-<sha>   │
          └─────────────────────┘
```

### 使用するMakeターゲット
- `make sync EXTRA=kaggle` - Kaggle依存関係のインストール
- `make kaggle-setup` - Kaggle認証情報の設定
- `make kaggle-push-dataset MSG=...` - コードをKaggle Datasetにpush
- `make gcloud-build TAG=...` - Cloud BuildでDockerイメージをビルド
- `make gcp-train TAG=...` - Vertex AIで訓練ジョブを投入
- `make gcp-download-model MODEL=...` - GCSからモデルをダウンロード
- `make kaggle-inference-submit MSG=...` - 推論を実行してKaggleに提出

---

## Manual Kaggle Submit (submit-manual.yaml)

### トリガー
- 手動実行のみ（`workflow_dispatch`）

### 入力パラメータ
| パラメータ | 必須 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `model_name` | Yes | `latest` | 使用するモデル名 |
| `message` | No | `Manual submission` | 提出メッセージ |

### 実行内容

```
┌─────────────────────────────────────────────────┐
│  submit job                                     │
├─────────────────────────────────────────────────┤
│  1. make sync EXTRA=kaggle                      │
│  2. make kaggle-setup                           │
│  3. GCP認証                                     │
│  4. make gcp-download-model MODEL=<model_name>  │
│  5. make kaggle-inference-submit MSG=<message>  │
└─────────────────────────────────────────────────┘
```

### 使用するMakeターゲット
- `make sync EXTRA=kaggle` - Kaggle依存関係のインストール
- `make kaggle-setup` - Kaggle認証情報の設定
- `make gcp-download-model MODEL=...` - GCSからモデルをダウンロード
- `make kaggle-inference-submit MSG=...` - 推論を実行してKaggleに提出

---

## 必要なSecrets/Variables

### Secrets
| 名前 | 説明 |
|-----|------|
| `KAGGLE_JSON` | Kaggle APIのJSONクレデンシャル |

### Variables
| 名前 | 説明 |
|-----|------|
| `GCP_PROJECT_ID` | GCPプロジェクトID |
| `GCP_REGION` | GCPリージョン（例: `asia-northeast1`） |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Workload Identity Federationプロバイダ |
| `GCP_SERVICE_ACCOUNT` | GCPサービスアカウント |
| `KAGGLE_COMPETITION` | Kaggleコンペティション名 |
| `KAGGLE_USERNAME` | Kaggleユーザー名 |

---

## Makefile設計

GitHub ActionsではすべてのコマンドがMakefileを経由して実行されます。

### CI環境での動作
`CI=true`環境変数が設定されている場合、`devbox run --`プレフィックスが省略され、直接`uv run`が実行されます。

```makefile
# CI環境ではdevboxを使わない
DEVBOX := $(if $(CI),,devbox run --)
```

### ローカルとCIの対応

| ローカル | CI |
|---------|-----|
| `devbox run -- uv run pytest` | `uv run pytest` |

これにより、ローカル開発とCI環境で同一のMakeターゲットを使用できます。

