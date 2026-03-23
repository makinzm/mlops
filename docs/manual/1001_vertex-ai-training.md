# Vertex AI 学習ガイド

前提条件: [docs/manual/1000_gcp-initial-setup.md](./1000_gcp-initial-setup.md) が完了済みであること。

> **このマニュアルのコンペティション固有の値について:**
>
> 本マニュアルの例は `titanic` コンペティションを前提にしています。
> これは `conf/config.yaml` の `defaults` で `competition: titanic` が設定されているためです。
> 別のコンペティションに切り替える場合は CLI で `competition=other_name` を渡すか、
> `conf/config.yaml` のデフォルトを変更してください。
> コンペティション固有の設定は `conf/competition/{name}/` 以下に配置されています。

---

## 1. Terraform でリソースを作成する

### terraform.tfvars の生成

`.env` から自動生成します（手動編集は不要）:

```bash
./scripts/gen_tfvars.sh
```

| ファイル | 役割 |
|---------|------|
| `.env` | GCP 設定の single source of truth |
| `scripts/gen_tfvars.sh` | `.env` → `terraform/terraform.tfvars` 変換スクリプト |
| `terraform/terraform.tfvars` | 自動生成（Git 管理外） |

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

| ファイル | 役割 |
|---------|------|
| `terraform/main.tf` | プロバイダー + モジュール定義 |
| `terraform/variables.tf` | 変数定義（project_id, region, bucket_name） |
| `terraform/modules/vertex_training/main.tf` | GCS, AR, SA, IAM のリソース定義 |

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

| ファイル | 役割 |
|---------|------|
| `conf/usecase/preprocess.yaml` | 前処理 usecase の基本設定 |
| `conf/competition/titanic/preprocess/base.yaml` | 前処理レシピ（`titanic` は `conf/config.yaml` のデフォルト competition） |

### Vertex AI で学習を実行

```bash
uv run python -m src usecase=vertex_train recipe=lgbm
```

| ファイル | 役割 |
|---------|------|
| `conf/usecase/vertex_train.yaml` | vertex_train usecase の設定 |
| `conf/gcp/vertex.yaml` | GCP プロジェクト・リージョン・マシンタイプ等（`.env` から読む） |
| `conf/competition/titanic/training/lgbm.yaml` | LightGBM ハイパーパラメータ（`titanic` は `conf/config.yaml` のデフォルト competition） |

このコマンドの内部処理:

```
[ローカル]
1. src/ + conf/ + scripts/ を GCS にアップロード（数秒）
2. 前処理済みデータを GCS にアップロード
3. Vertex AI に CustomJob を送信
        ↓
[Vertex AI コンテナ（Kaggle イメージ）]
4. Python SDK でコードを GCS から /app/ にダウンロード
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

**変更ファイル: `conf/gcp/vertex.yaml`**

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

## 4. Kaggle に提出する

Vertex AI で学習したモデルを Kaggle Notebook で推論・提出する手順です。
既存の `mlops-pipeline-src` Dataset / `titanic-pipeline` Notebook には触りません。

### 4-1. 推論（ローカルで submission.csv を生成）

```bash
uv run python -m src usecase=inference recipe=titanic_ensemble
```

| ファイル | 役割 |
|---------|------|
| `conf/usecase/inference.yaml` | 推論 usecase の基本設定 |
| `conf/competition/titanic/inference/titanic_ensemble.yaml` | モデルパス・特徴量・アンサンブル戦略（`titanic` は `conf/config.yaml` のデフォルト competition） |

### 4-2. コード（src + conf）を Kaggle Dataset に push

```bash
uv run python -m src usecase=update_source_dataset
```

| ファイル | 役割 |
|---------|------|
| `conf/usecase/update_source_dataset.yaml` | Dataset slug (`mlops-pipeline-src`)、アップロード対象ディレクトリ |

### 4-3. 最新モデル重みを Kaggle Dataset に push

最新タイムスタンプのモデルだけが自動選択されます（`latest` 解決）。

```bash
# 初回（Dataset 作成）
uv run python -m src usecase=create_vertex_models

# 2回目以降（バージョン更新）
uv run python -m src usecase=upload_vertex_models
```

| ファイル | 役割 | 主な設定項目 |
|---------|------|------------|
| `conf/usecase/create_vertex_models.yaml` | Dataset 初回作成 | `dataset_slug`, `src_dir` |
| `conf/usecase/upload_vertex_models.yaml` | Dataset バージョン更新 | 同上 |

### 4-4. 推論専用 Notebook を push

```bash
uv run python -m src usecase=push_vertex_notebook
```

| ファイル | 役割 | 主な設定項目 |
|---------|------|------------|
| `conf/usecase/push_vertex_notebook.yaml` | Notebook push 設定 | `kernel_slug`, `recipe`, `extra_datasets` |
| `conf/competition/titanic/pipeline/inference_only.yaml` | 推論のみパイプライン（`titanic` は `conf/config.yaml` のデフォルト competition） |
| `templates/notebook/pipeline.ipynb.j2` | Notebook テンプレート |

### 4-5. フル自動パイプライン（前処理〜提出まで一括）

上記 4-1 〜 4-4 を一括実行:

```bash
uv run python -m src usecase=pipeline recipe=vertex_to_kaggle
```

| ファイル | 役割 |
|---------|------|
| `conf/competition/titanic/pipeline/vertex_to_kaggle.yaml` | パイプライン定義（`titanic` は `conf/config.yaml` のデフォルト competition） |

パイプライン内容:

| Step | 内容 | 使用 config |
|------|------|------------|
| ① preprocess | 前処理 | `conf/competition/{name}/preprocess/base.yaml` |
| ② vertex_train | Vertex AI で学習 | `conf/competition/{name}/training/lgbm.yaml` |
| ③ inference | 推論（submission.csv 生成） | `conf/competition/{name}/inference/titanic_ensemble.yaml` |
| ④ update_source_dataset | コードを `mlops-pipeline-src` に push | `conf/usecase/update_source_dataset.yaml` |
| ⑤ update_source_dataset | モデルを `titanic-vertex-models` に push | ⑤ は pipeline yaml 内で直接定義 |
| ⑥ push_notebook | `titanic-vertex-inference` を push（推論のみ） | ⑥ は pipeline yaml 内で直接定義 |

---

## 5. カスタム Docker イメージを使う場合（上級者向け）

Kaggle プリビルトイメージで不足するパッケージがある場合、
自前のイメージをビルド・push できます。

```bash
# ビルド + push（.env から GCP_PROJECT / GCP_REGION を読む）
./scripts/docker_push.sh
```

| ファイル | 役割 |
|---------|------|
| `Dockerfile` | カスタムイメージ定義 |
| `scripts/docker_push.sh` | ビルド + push スクリプト |

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
