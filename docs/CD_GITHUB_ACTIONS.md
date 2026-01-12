# GitHub ActionsでのCD実行

GitHub Actionsを使用したCD設定手順です。

---

## 前提条件

- GCPプロジェクトが作成済み（[CD_LOCAL.md](./CD_LOCAL.md) のセクション1参照）
- Kaggle APIトークンが取得済み

---

## 1. Workload Identity Federation設定

GitHub ActionsからGCPに安全に認証するための設定です。サービスアカウントキーを使わずにOIDCで認証します。

### 1.1 Workload Identity Pool作成

```bash
# Pool作成
gcloud iam workload-identity-pools create "github-pool" \
  --location="global" \
  --display-name="GitHub Actions Pool"

# Provider作成
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
  --issuer-uri="https://token.actions.githubusercontent.com"
```

### 1.2 サービスアカウントへの権限付与

```bash
PROJECT_ID=$(gcloud config get-value project)
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
SA_EMAIL="mlops-cicd@${PROJECT_ID}.iam.gserviceaccount.com"

# GitHub Actionsからのなりすましを許可
# YOUR_GITHUB_ORG/YOUR_REPO を実際のリポジトリに置き換え
gcloud iam service-accounts add-iam-policy-binding $SA_EMAIL \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/attribute.repository/YOUR_GITHUB_ORG/YOUR_REPO"
```

### 1.3 Workload Identity Provider名の取得

GitHub Actionsで使用するProvider名を取得します：

```bash
gcloud iam workload-identity-pools providers describe github-provider \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --format="value(name)"
```

出力例：
```
projects/123456789/locations/global/workloadIdentityPools/github-pool/providers/github-provider
```

---

## 2. GitHub Secrets/Variables設定

GitHubリポジトリの Settings → Secrets and variables → Actions で設定します。

### Secrets

| 名前 | 値 | 説明 |
|-----|-----|------|
| `KAGGLE_JSON` | `{"username":"xxx","key":"xxx"}` | Kaggle API認証情報（kaggle.jsonの中身） |

### Variables

| 名前 | 値の例 | 説明 |
|-----|-------|------|
| `GCP_PROJECT_ID` | `your-project-id` | GCPプロジェクトID |
| `GCP_REGION` | `asia-northeast1` | GCPリージョン |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | `projects/123456/locations/global/workloadIdentityPools/github-pool/providers/github-provider` | 1.3で取得した値 |
| `GCP_SERVICE_ACCOUNT` | `mlops-cicd@your-project-id.iam.gserviceaccount.com` | サービスアカウントメール |
| `KAGGLE_COMPETITION` | `your-competition-name` | Kaggleコンペティション名 |
| `KAGGLE_USERNAME` | `your-kaggle-username` | Kaggleユーザー名 |

---

## 3. ワークフロー実行方法

### 3.1 自動実行（Push時）

`main`ブランチにPushすると自動的にCDパイプラインが実行されます。

```bash
git push origin main
```

### 3.2 手動実行

GitHub上で手動実行する場合：

1. リポジトリの「Actions」タブを開く
2. 左メニューから「CD - Deploy and Train」を選択
3. 「Run workflow」ボタンをクリック
4. オプション設定：
   - `skip_training`: チェックするとVertex AI訓練をスキップ
5. 「Run workflow」をクリック

### 3.3 特定モデルで手動提出

1. 「Actions」タブから「Manual Kaggle Submit」を選択
2. 「Run workflow」ボタンをクリック
3. パラメータ設定：
   - `model_name`: 使用するモデル名（デフォルト: `latest`）
   - `message`: 提出メッセージ
4. 「Run workflow」をクリック

---

## 4. ワークフロー構成

```
┌─────────────────────┐     ┌─────────────────────┐
│  kaggle-push job    │     │  vertex-train job   │
│  (並列実行)         │     │  (並列実行)         │
├─────────────────────┤     ├─────────────────────┤
│ 1. Kaggle依存関係   │     │ 1. GCP認証          │
│ 2. Kaggle認証設定   │     │ 2. Cloud Build      │
│ 3. Dataset Push     │     │ 3. Vertex AI訓練    │
└─────────┬───────────┘     └──────────┬──────────┘
          │                            │
          └──────────┬─────────────────┘
                     ▼
          ┌─────────────────────┐
          │  kaggle-submit job  │
          │  (両方完了後)       │
          ├─────────────────────┤
          │ 1. モデルダウンロード │
          │ 2. 推論実行         │
          │ 3. Kaggle提出       │
          └─────────────────────┘
```

---

## 5. トラブルシューティング

### Workload Identity認証エラー

```
Error: google-github-actions/auth failed with:
the caller does not have permission
```

**確認事項：**
1. リポジトリ名が正しく設定されているか
   ```bash
   # 現在の設定確認
   gcloud iam service-accounts get-iam-policy $SA_EMAIL
   ```

2. attribute.repositoryの形式が正しいか
   - 正: `YOUR_ORG/YOUR_REPO`（スラッシュ区切り）
   - 誤: `YOUR_ORG-YOUR_REPO`（ハイフン区切り）

### Variables/Secretsが見つからない

```
Error: vars.GCP_PROJECT_ID is not defined
```

**確認事項：**
- リポジトリのSettings → Secrets and variables → Actionsで設定されているか
- 「Variables」タブで設定しているか（「Secrets」タブではない）

### Kaggle認証エラー

```
401 - Unauthorized
```

**確認事項：**
- `KAGGLE_JSON`シークレットがJSON形式で正しく設定されているか
- `kaggle.json`ファイルの中身をそのままコピーしているか

---

## 6. セキュリティベストプラクティス

1. **サービスアカウントキーは使わない**: Workload Identity Federationを使用
2. **最小権限の原則**: 必要なロールのみ付与
3. **リポジトリ制限**: 特定リポジトリからのみ認証を許可
4. **Secretsの管理**: API認証情報はSecrets、設定値はVariablesに分離
