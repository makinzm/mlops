# Timeline: Vertex AI Training 実装

## 2026-03-22

### Session 1: 計画策定

1. プロジェクト全体の構造を調査（src, conf, tests, mille.toml）
2. 実装計画を `README.md` に作成
3. ユーザーからの追加要件:
   - Kaggle Notebook への自動反映パイプライン
   - コスト監視・予算上限時のジョブ停止/警告切り替え
   - IAM 権限設計
4. 計画承認

### Session 2: RED → GREEN 実装

#### RED phase (テスト実装)

テストを先に実装し、`--no-verify` でコミット:

- `tests/infrastructure/gcp/test_storage.py` — GCSRepositoryImpl のモックテスト
- `tests/infrastructure/gcp/test_vertex_ai.py` — VertexAIRepositoryImpl のモックテスト
- `tests/usecase/training/test_vertex_train_usecase.py` — VertexAITrainUseCase の全フローテスト
- `tests/usecase/pipeline/test_pipeline_vertex.py` — Pipeline の vertex_train ステップテスト

commit: `c62eb6b test(RED): GCSRepository/VertexAI/VertexAITrainUseCase/pipeline vertex steps のテスト実装`

#### GREEN phase (実装)

##### Domain Protocol
- `src/domain/repository/gcs.py` — GCSRepository Protocol
- `src/domain/repository/vertex_ai.py` — VertexAIRepository Protocol, VertexJobStatus

##### Infrastructure
- `src/infrastructure/gcp/storage.py` — GCSRepositoryImpl (google-cloud-storage)
- `src/infrastructure/gcp/vertex_ai.py` — VertexAIRepositoryImpl (google-cloud-aiplatform)

##### UseCase
- `src/usecase/training/vertex_train.py` — VertexAITrainUseCase, VertexTrainResult

##### Pipeline 拡張
- `src/usecase/pipeline/pipeline.py` — `**extra_runners` で拡張可能に
- `src/main.py` — vertex_train / update_source_dataset / push_notebook runner 追加

##### Config
- `conf/gcp/vertex.yaml` — GCP 固有設定
- `conf/usecase/vertex_train.yaml` — vertex_train usecase
- `conf/competition/titanic/pipeline/vertex_to_kaggle.yaml` — 全自動パイプライン
- `conf/competition/titanic/training/lgbm.yaml` — `${oc.env:...}` で env var オーバーライド対応
- `conf/config.yaml` — gcp, job_id パラメータ追加

##### Terraform
- `terraform/main.tf` — Provider + Module
- `terraform/variables.tf` — 変数定義
- `terraform/outputs.tf` — 出力値
- `terraform/modules/vertex_training/main.tf` — 全リソース定義
- `terraform/modules/vertex_training/budget_enforcer/main.py` — Cloud Function

##### Docker
- `Dockerfile` — 学習コンテナ
- `scripts/vertex_entrypoint.py` — コンテナ内エントリーポイント

##### マニュアル
- `docs/manual/1000_gcp-initial-setup.md` — GCP 初心者向けセットアップ
- `docs/manual/1001_vertex-ai-training.md` — Vertex AI 学習ガイド
- `docs/manual/1002_cost-monitoring.md` — コスト監視ガイド

### 検証結果

- 全 343 テスト PASS
- mille check: 0 violations
- ruff check: PASS
- ruff format: PASS
- mypy: PASS (strict)
