# 実行環境の切り替えガイド

## 概要

このプロジェクトの学習処理は以下の環境で実行できる。

| 環境 | usecase | 前提 |
|------|---------|------|
| ローカル（CPU） | `usecase=train` | `uv sync` のみ |
| ローカル（GPU） | `usecase=train` | CUDA 環境 + `uv sync` |
| GCP Vertex AI（非同期） | `usecase=vertex_submit` | GCP セットアップ済み |
| GCP Vertex AI（同期） | `usecase=remote_train` | GCP セットアップ済み |
| AWS | 未実装 | — |

---

## 1. ローカル実行

### 設定

変更する設定ファイル: `conf/config.yaml`

```yaml
defaults:
  - competition: titanic   # コンペ名を変更する
  - usecase: train
```

コンペ固有の学習レシピ: `conf/competition/{name}/training/{recipe}.yaml`

### 実行コマンド

```bash
# デフォルトレシピで学習
uv run python -m src usecase=train competition=titanic

# レシピを指定して学習
uv run python -m src usecase=train competition=titanic recipe=lgbm

# AudioTrainer を使う例（音声コンペ）
uv run python -m src usecase=train competition=audio_example recipe=efficientnet_b0
```

### GPU の自動検出

`uv run python -m src usecase=train ...` 実行時、PyTorch が CUDA を自動検出して GPU を使用する。
明示的に確認する場合:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

---

## 2. GCP Vertex AI 実行

GCP の初期セットアップは [docs/manual/1000_gcp-initial-setup.md](./1000_gcp-initial-setup.md) を参照。

### 設定

`.env` ファイルに以下を設定する:

```
GCP_PROJECT=your-project-id
GCP_REGION=asia-northeast1
GCP_BUCKET_NAME=your-bucket-name
```

`conf/cloud/vertex.yaml` で machine_type と container_uri を確認する:

```yaml
cloud:
  machine_type: n1-standard-4       # CPU 実行
  accelerator_type: null            # GPU 使用時: NVIDIA_TESLA_T4
  accelerator_count: 0              # GPU 使用時: 1
  container_uri: gcr.io/kaggle-images/python:latest  # CPU
  # GPU の場合: gcr.io/kaggle-gpu-images/python
```

### GPU を使う場合の設定変更

`conf/cloud/vertex.yaml` を直接編集するか、CLI で上書きする:

```bash
# T4 GPU 1枚で実行する例
uv run python -m src usecase=vertex_submit competition=titanic recipe=lgbm \
  cloud.machine_type=n1-standard-4-t4 \
  cloud.accelerator_type=NVIDIA_TESLA_T4 \
  cloud.accelerator_count=1 \
  cloud.container_uri=gcr.io/kaggle-gpu-images/python
```

### 実行コマンド

#### 非同期送信（推奨: ジョブを送信して即終了）

```bash
uv run python -m src usecase=vertex_submit competition=titanic recipe=lgbm
```

ジョブIDが返るので、完了後に以下でモデルをダウンロードする:

```bash
uv run python -m src usecase=vertex_download competition=titanic
```

#### 同期実行（ジョブ完了まで待機）

```bash
uv run python -m src usecase=remote_train competition=titanic recipe=lgbm
```

---

## 3. executor 設定について（preprocess ステップ）

`conf/competition/{name}/preprocess/{recipe}.yaml` に `executor:` を設定できる。

```yaml
executor:
  type: local    # 唯一の実装済み設定
```

`conf/executor/gcp_vertex.yaml` と `conf/executor/ray_local.yaml` は将来用のプレースホルダーであり、
現在設定すると `NotImplementedError` が発生する。preprocess の分散実行は未実装。

---

## 4. AWS について

AWS での実行は未実装。設計判断: Vertex AI を主要クラウドとし、AWS 対応は需要が生じた時点で検討する。
現時点では AWS サポートの ADR は作成しない（先行実装を避ける原則）。
