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

## 2. 学習の実行

**Docker のビルドや push は不要です。**
Kaggle が公開しているプリビルトイメージ（ML ライブラリが全部入り）をそのまま使います。
コード（`src/`, `conf/`, `scripts/`）は実行時に自動で GCS にアップロードされ、
コンテナ内で取得されます。

### 前処理を実行（初回のみ、またはデータ変更時）

```bash
uv run python -m src usecase=preprocess recipe=base
```

### Vertex AI で学習を実行

```bash
uv run python -m src usecase=vertex_train recipe=lgbm
```

このコマンドの内部処理:

```
[ローカル]
1. src/ + conf/ + scripts/ を GCS にアップロード（数秒）
2. 前処理済みデータを GCS にアップロード
3. Vertex AI に CustomJob を送信
        ↓
[Vertex AI コンテナ（Kaggle イメージ）]
4. gsutil でコードを /app/ にダウンロード
5. pip install で不足 deps をインストール（hydra-core, omegaconf 等）
6. 学習を実行
7. モデルを GCS にアップロード
        ↓
[ローカル]
8. GCS からモデルをダウンロード → models/titanic/titanic_lgbm/ に保存
```

### ジョブの確認方法（GCP コンソール）

1. GCP コンソール → 左上メニュー → 「Vertex AI」
2. 「トレーニング」→「カスタムジョブ」
3. ジョブのステータスやログを確認できます

---

## 3. CPU / GPU の切り替え

`conf/gcp/vertex.yaml` の `container_uri` と `machine_type` を変更します。

### CPU（デフォルト）

```yaml
gcp:
  container_uri: gcr.io/kaggle-images/python:latest
  machine_type: n1-standard-4    # vCPU 4 / RAM 15GB
  accelerator_type: null
  accelerator_count: 0
```

### GPU

```yaml
gcp:
  container_uri: gcr.io/kaggle-gpu-images/python:latest
  machine_type: n1-standard-4
  accelerator_type: NVIDIA_TESLA_T4
  accelerator_count: 1
```

> **GPU の料金目安:**
>
> | GPU | 目安コスト/時間 |
> |-----|----------------|
> | NVIDIA_TESLA_T4 | 約 $0.35 |
> | NVIDIA_TESLA_V100 | 約 $2.48 |
> | NVIDIA_TESLA_A100 | 約 $3.67 |
>
> LightGBM は CPU で十分高速です。GPU は PyTorch / TensorFlow の学習時に使ってください。

---

## 4. フル自動パイプラインの実行

前処理 → Vertex AI 学習 → 推論 → Kaggle Dataset 更新 → Notebook 更新 を一括実行:

```bash
uv run python -m src usecase=pipeline recipe=vertex_to_kaggle
```

このパイプラインは `conf/competition/titanic/pipeline/vertex_to_kaggle.yaml` で定義されています。

---

## 5. カスタム Docker イメージを使う場合（上級者向け）

Kaggle プリビルトイメージで不足するパッケージがある場合、
自前のイメージをビルド・push できます。

```bash
# ビルド + push（.env から GCP_PROJECT / GCP_REGION を読む）
./scripts/docker_push.sh
```

push 後、`conf/gcp/vertex.yaml` の `container_uri` を変更:

```yaml
gcp:
  container_uri: ${oc.env:GCP_REGION,asia-northeast1}-docker.pkg.dev/${oc.env:GCP_PROJECT}/mlops/training:latest
```

> カスタムイメージの再ビルドが必要なのは `pyproject.toml` / `uv.lock` を変更したときだけです。
> コード変更時は再ビルド不要（GCS 経由で渡されるため）。

---

## 注意事項

- 学習ジョブの実行中はコストが発生します。コスト確認・予算アラートの設定方法は
  [docs/manual/1002_cost-monitoring.md](./1002_cost-monitoring.md) を参照してください
- `terraform.tfvars` は `.gitignore` で除外されています。Git にコミットしないこと
