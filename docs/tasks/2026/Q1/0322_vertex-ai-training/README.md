# Vertex AI Training 実装計画

## 概要

ローカルで動く LightGBM 学習パイプラインを GCP Vertex AI で実行し、
学習済みモデルを自動的に Kaggle Notebook に反映して提出できるようにする。

コスト監視・上限アラートもセットで実装する。

---

## エンドゴール

```bash
# 1コマンドで「前処理 → Vertex AI 学習 → 推論 → Kaggle Notebook 更新」が走る
uv run python -m src usecase=pipeline recipe=vertex_to_kaggle
```

---

## 完全なデータフロー

```
[ローカル]
  uv run python -m src usecase=vertex_train recipe=lgbm
         │
         │ 1. preprocessed data を GCS にアップロード
         ▼
[GCS: gs://{bucket}/staging/{job_id}/data/]
         │
         │ 2. Vertex AI CustomJob に送信
         ▼
[Vertex AI コンテナ]
  ├── GCS からデータを /tmp/data/ にダウンロード
  ├── uv run python -m src usecase=train recipe=lgbm
  └── models/ を GCS にアップロード
         │
         │ 3. ジョブ完了をポーリング（ログをリアルタイム表示）
         ▼
[GCS: gs://{bucket}/staging/{job_id}/models/]
         │
         │ 4. ローカルに models/ をダウンロード
         ▼
[ローカル: models/titanic/{job_id}/{timestamp}/]
         │
         │ 5. Pipeline の次ステップへ
         ▼
  ├── inference (submission.csv 生成)
  ├── update_source_dataset (モデルを Kaggle Dataset に push)
  └── push_notebook (Kaggle Notebook を更新)
```

---

## 実装スコープ

### Phase 1: Terraform — GCP インフラ定義

```
terraform/
├── main.tf              - プロバイダー + バックエンド設定
├── variables.tf         - 変数定義（project, region, budget_amount 等）
├── outputs.tf           - 出力値（bucket名, SA メール等を conf に貼り付けやすく）
└── modules/
    └── vertex_training/
        ├── main.tf      - リソース定義
        └── variables.tf
```

**作成リソース:**

| リソース | 用途 |
|---------|------|
| `google_storage_bucket` | データ・モデルのステージング |
| `google_artifact_registry_repository` | Docker イメージ保管 |
| `google_service_account` (training-sa) | Vertex AI ジョブ実行用 SA |
| `google_service_account` (reader-sa) | Kaggle Notebook からモデル読み取り専用 |
| `google_project_iam_member` × 4 | 最小権限の IAM バインディング |
| `google_billing_budget` | コスト上限アラート |
| `google_pubsub_topic` | Budget → Cloud Function への通知経路 |
| `google_cloudfunctions2_function` | 予算超過時に Vertex AI ジョブを停止 |

**IAM 設計（最小権限原則）:**

```
training-sa (Vertex AI ジョブが使うアカウント)
  └── roles/storage.objectAdmin       → GCS の読み書き
  └── roles/aiplatform.user           → 自分自身のジョブ操作
  └── roles/logging.logWriter         → Cloud Logging への書き込み
  └── roles/artifactregistry.reader   → Docker イメージの pull

reader-sa (Kaggle Notebook / ローカル推論が使うアカウント)
  └── roles/storage.objectViewer      → GCS からモデルを読む（書き込み不可）

開発者 (gcloud auth で認証する自分)
  └── roles/aiplatform.user           → ジョブ送信・監視
  └── roles/storage.objectAdmin       → GCS のアップロード・ダウンロード
  └── roles/artifactregistry.writer   → Docker イメージの push
```

**コスト管理 (google_billing_budget):**

```
budget_amount: 設定可能（変数で管理）
threshold_rules:
  - threshold_percent: 0.5   → 50%  で EMAIL 通知
  - threshold_percent: 0.8   → 80%  で EMAIL 通知
  - threshold_percent: 1.0   → 100% で EMAIL + Pub/Sub (→ Cloud Function)
  - threshold_percent: 1.2   → 120% で EMAIL + Pub/Sub (→ Cloud Function)

Cloud Function の動作モード (variables.tf で切り替え):
  budget_action = "warn"   → Pub/Sub メッセージを受け取っても通知のみ
  budget_action = "stop"   → 実行中の Vertex AI ジョブを全停止
```

### Phase 2: Docker イメージ

```
Dockerfile
scripts/
└── vertex_entrypoint.sh   - コンテナ内で動くスクリプト
```

`vertex_entrypoint.sh` の処理:
1. `gsutil -m cp -r {GCS_DATA_URI} /tmp/data/` でデータを取得
2. Hydra の `data_dir` を環境変数でオーバーライドして `uv run python -m src usecase=train` を実行
3. `gsutil -m cp -r /tmp/models/ {GCS_MODEL_URI}` でモデルを保存

### Phase 3: Python コード

```
src/domain/repository/
├── gcs.py            - GCSRepository Protocol (upload_dir, download_dir)
└── vertex_ai.py      - VertexAIRepository Protocol (submit_custom_job, wait_for_job)

src/infrastructure/gcp/
├── __init__.py
├── storage.py        - GCSRepositoryImpl (google-cloud-storage)
└── vertex_ai.py      - VertexAIRepositoryImpl (google-cloud-aiplatform)

src/usecase/training/
└── vertex_train.py   - VertexAITrainUseCase

conf/gcp/
└── vertex.yaml       - project, region, bucket, container_uri, machine_type 等

conf/usecase/
└── vertex_train.yaml - usecase: vertex_train

conf/competition/titanic/pipeline/
└── vertex_to_kaggle.yaml  - 新しいパイプライン設定
```

**VertexAITrainUseCase の処理:**

```python
def execute(self) -> VertexTrainResult:
    # 1. preprocessed data を GCS にアップロード
    # 2. CustomJob を送信（container_uri, args, env_vars, machine_type）
    # 3. ジョブ完了をポーリング（ログをリアルタイム Streaming 表示）
    # 4. GCS からモデルをダウンロード → ローカルの models/ に配置
    # 5. VertexTrainResult を返す（TrainResult と互換の .job_id を持つ）
```

### Phase 4: パイプライン設定 (Kaggle 提出まで自動化)

`conf/competition/titanic/pipeline/vertex_to_kaggle.yaml`:

```yaml
steps:
  - usecase: preprocess
    recipe: base
  - usecase: vertex_train        # Vertex AI で学習
    recipe: lgbm
  - usecase: inference           # ローカルで推論 (submission.csv 生成)
    recipe: titanic_ensemble
  - usecase: update_source_dataset  # モデル + コードを Kaggle Dataset に push
    source_dataset.version_message: "auto: ${job_id}"
  - usecase: push_notebook       # Kaggle Notebook を最新モデルで更新
    notebook: titanic
```

### Phase 5: マニュアル (GCP 完全初心者向け)

```
docs/manual/
├── 1000_gcp-initial-setup.md      - GCP アカウント作成〜認証 (完全初心者向け)
├── 1001_vertex-ai-training.md     - Vertex AI 学習の実行方法
└── 1002_cost-monitoring.md        - コスト監視・上限設定の管理方法
```

**1000_gcp-initial-setup.md の内容:**
1. Google アカウント・GCP コンソールへのアクセス
2. プロジェクト作成（スクリーンショット付き手順）
3. 課金アカウントの有効化（無料枠 $300 の説明も）
4. 必要な API の有効化（Vertex AI, Cloud Storage, Artifact Registry, Cloud Functions, Pub/Sub, Billing）
5. `gcloud` CLI のインストール・初期設定
6. `gcloud auth application-default login` で認証
7. Terraform のインストール（tfenv 経由）
8. `terraform apply` でリソース作成
9. Docker イメージのビルド・push
10. `.env` への GCP 設定の記入

**1002_cost-monitoring.md の内容:**
- GCP コンソールでのリアルタイムコスト確認方法
- Budget アラートメールの確認方法
- `budget_action = "warn"` vs `"stop"` の切り替え方
- Vertex AI ジョブの手動停止方法（緊急時）
- 月末に `terraform destroy` でリソース削除する手順（使い終わったら消す）

---

## 設定ファイル設計

### conf/gcp/vertex.yaml

```yaml
# @package _global_
gcp:
  project: my-gcp-project                              # GCP プロジェクト ID
  region: asia-northeast1                              # 東京リージョン
  staging_bucket: gs://my-project-mlops-staging       # Terraform で作成
  container_uri: asia-northeast1-docker.pkg.dev/my-project/mlops/training:latest
  machine_type: n1-standard-4                         # vCPU 4 / 15GB RAM
  accelerator_type: null                              # GPU使用時: NVIDIA_TESLA_T4
  accelerator_count: 0
  service_account: training-sa@my-project.iam.gserviceaccount.com
  budget_action: warn                                 # "warn" or "stop"
```

---

## DoD チェックリスト

- [ ] Red → Green → Refactor のコミット粒度
- [ ] テストが lefthook / GitHub Actions で自動実行
- [ ] 手動確認手順が `docs/manual/` に記載
- [ ] 全変数が Hydra config で管理（GCP 設定含む）
- [ ] output_dir に per-directory .gitignore を動的生成
- [ ] output_dir に README.md を生成
- [ ] commit hash を train_result.yaml に記録
- [ ] seed で再現性を保証
- [ ] IAM 最小権限で設定
- [ ] Budget アラート設定済み

---

## 作業ステップ

1. [x] feature branch 作成 (`feat/vertex-ai-training`)
2. [x] この計画ドキュメント作成
3. [ ] **承認待ち**
4. [ ] lefthook/CI でテスト自動実行を確認
5. [ ] テスト実装 (RED) + TEST_LOG に記録
6. [ ] 実装 (GREEN)
7. [ ] リファクタ
8. [ ] マニュアル作成
9. [ ] PR 作成
