# Vertex AI 学習ガイド

前提条件: [docs/manual/1000_gcp-initial-setup.md](./1000_gcp-initial-setup.md) が完了済みであること。

---

## 1. Terraform でリソースを作成する

### terraform.tfvars の生成

`.env` から自動生成します（手動編集は不要）:

```bash
./scripts/gen_tfvars.sh
```

生成された `terraform/terraform.tfvars` の内容を確認:

```bash
cat terraform/terraform.tfvars
```

> `.env` が single source of truth です。値を変えたいときは `.env` を編集して再度 `./scripts/gen_tfvars.sh` を実行してください。

### Terraform の実行

```bash
cd terraform

# 初期化（初回のみ）
terraform init

# 変更内容の確認
terraform plan

# リソースの作成（確認後 yes を入力）
terraform apply
```

> **`terraform init/plan/apply` でエラーが出た場合:**
>
> | エラーメッセージ | 対処 |
> |---|---|
> | `No credentials loaded` | `gcloud auth application-default login` を実行 |
> | `requires a quota project` | `gcloud auth application-default set-quota-project YOUR_PROJECT_ID` を実行 |
> | `Inconsistent dependency lock file` | `terraform init -upgrade` を実行（`.tf` ファイル変更後に必要） |
> | `API has not been used ... or it is disabled` | [1000_gcp-initial-setup.md](./1000_gcp-initial-setup.md) セクション 7 の API 有効化を再確認。`gcloud services enable <api名>` で個別に有効化 |
>
> `terraform destroy` → `terraform apply` でやり直す場合も、先に `terraform init -upgrade` を実行してください。

### 出力値の確認

`terraform apply` 完了後、以下の出力値が表示されます:

```
staging_bucket_uri = "gs://your-project-mlops-staging"
container_registry_uri = "asia-northeast1-docker.pkg.dev/your-project/mlops"
training_service_account_email = "training-sa@your-project.iam.gserviceaccount.com"
reader_service_account_email = "reader-sa@your-project.iam.gserviceaccount.com"
```

> `conf/gcp/vertex.yaml` の手動更新は不要です。
> `.env` の `GCP_PROJECT` / `GCP_REGION` から `${oc.env:...}` で自動的に値が解決されます。

---

## 2. Docker イメージのビルドとプッシュ

### Docker の認証設定

```bash
gcloud auth configure-docker asia-northeast1-docker.pkg.dev
```

### イメージのビルド

```bash
cd /path/to/mlops  # プロジェクトルート

docker build -t asia-northeast1-docker.pkg.dev/your-project/mlops/training:latest .
```

### イメージのプッシュ

```bash
docker push asia-northeast1-docker.pkg.dev/your-project/mlops/training:latest
```

---

## 3. 初回動作確認

### 前処理を実行（データが必要）

```bash
uv run python -m src usecase=preprocess recipe=base
```

### Vertex AI で学習を実行

```bash
uv run python -m src usecase=vertex_train recipe=lgbm
```

このコマンドは以下の処理を行います:
1. 前処理済みデータを GCS にアップロード
2. Vertex AI CustomJob を送信
3. ジョブの完了を待機（数分〜数十分）
4. 学習済みモデルを GCS からローカルにダウンロード
5. モデルを `models/titanic/titanic_lgbm/` に保存

### ジョブの確認方法（GCP コンソール）

1. GCP コンソール → 左上メニュー → 「Vertex AI」
2. 「トレーニング」→「カスタムジョブ」
3. ジョブのステータスや logs を確認できます

---

## 4. フル自動パイプラインの実行

前処理 → Vertex AI 学習 → 推論 → Kaggle Dataset 更新 → Notebook 更新 を一括実行:

```bash
uv run python -m src usecase=pipeline recipe=vertex_to_kaggle
```

このパイプラインは `conf/competition/titanic/pipeline/vertex_to_kaggle.yaml` で定義されています。

---

## 注意事項

- `terraform.tfvars` には請求アカウント ID が含まれるため、絶対に Git にコミットしないこと
  （`.gitignore` で除外されています）
- 学習ジョブの実行中はコストが発生します。コスト確認方法は
  [docs/manual/1002_cost-monitoring.md](./1002_cost-monitoring.md) を参照してください
