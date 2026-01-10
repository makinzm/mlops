# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KaggleOpsを起点としたMLOpsプロジェクト。Clean Architectureに基づき、特定プラットフォーム（Kaggle/GCP等）に依存しない設計。

## Build and Development Commands

```bash
uv sync                    # 依存関係インストール
uv run pytest              # テスト実行
uv run python -m src.cli   # CLI実行
```

## Directory Structure

```
mlops/
├── src/
│   ├── core/                     # ドメイン層（外部依存なし）
│   │   ├── models/              # モデル定義（PyTorch/Lightning）
│   │   ├── features/            # 特徴量エンジニアリング
│   │   ├── metrics/             # 評価メトリクス
│   │   └── schemas/             # データスキーマ・バリデーション
│   │
│   ├── usecases/                # アプリケーション層
│   │   ├── eda.py               # 探索的データ分析
│   │   ├── create_dataset.py    # データセット作成
│   │   ├── train.py             # モデル学習
│   │   ├── evaluate.py          # モデル検証
│   │   ├── inference.py         # 推論
│   │   └── submit.py            # 提出（Model Servingの一形態）
│   │
│   ├── ports/                   # ポート（抽象インターフェース）
│   │   ├── data_store.py        # データ保存/読込
│   │   ├── model_store.py       # モデル保存/読込
│   │   ├── experiment_tracker.py # 実験追跡
│   │   └── serving_gateway.py   # モデル提供（Kaggle/API/Batch）
│   │
│   └── adapters/                # インフラ層（具象実装）
│       ├── local/               # ローカル実行
│       ├── kaggle/              # Kaggle API
│       ├── gcp/                 # GCP (Vertex AI, GCS)
│       └── mlflow/              # MLflow
│
├── notebooks/                   # EDA・実験用ノートブック
├── configs/                     # Hydra設定
│   ├── config.yaml
│   ├── model/
│   ├── data/
│   └── infra/                   # adapter切り替え設定
├── data/                        # データ（gitignore）
│   ├── raw/
│   ├── processed/
│   └── features/
├── outputs/                     # 出力（gitignore）
│   ├── models/
│   ├── predictions/
│   └── submissions/
├── tests/
└── scripts/
```

## Architecture Principles

1. **依存関係の方向**: adapters → ports ← usecases ← core
2. **core層は外部依存禁止**: PyTorch等のMLライブラリのみ許可
3. **ports層で抽象化**: `ServingGateway`がKaggle提出もAPI公開も同一インターフェースで扱う
4. **Hydraでadapter切り替え**: `infra=kaggle` or `infra=gcp` で実行環境を切り替え

## Workflow Mapping

```
EDA          → notebooks/, usecases/eda.py
Create Dataset → usecases/create_dataset.py, core/features/
Train Model  → usecases/train.py, core/models/
Check Model  → usecases/evaluate.py, core/metrics/
Model Serving → ports/serving_gateway.py
Inference    → usecases/inference.py
Submit       → usecases/submit.py (ServingGateway経由)
```

## MLOps DoD (Definition of Done) 概要

| # | 層 | DoD | 主要ツール |
|---|---|---|---|
| 0 | インフラ | 環境が再現可能・コード管理 | Terraform, Ansible, Helm |
| 1 | CI/CD | コード変更が自動テスト・デプロイ | uv, pytest, GitHub Actions, Docker, ArgoCD |
| 2 | 実験管理 | すべての実験が再現可能 | PyTorchLightning, PyTorch, Hydra, MLflow, DVC |
| 3 | データパイプライン | データ処理が自動化・スケーラブル | Airflow, dbt, Feast |
| 4 | モデル検証 | モデル品質が定量的に保証 | Great Expectations, pytest, Evidently, CrossValidation |
| 5 | デプロイメント | モデルが安全に本番稼働 | KServe, ONNX Runtime, Kubernetes |
| 6 | オブザーバビリティ | システム状態が常に可視化 | Prometheus, Grafana, OpenTelemetry |
| 7 | MLモニタリング | モデル劣化が早期検知 | Evidently, Alibi Detect |
| 8 | A/Bテスト | 新モデル効果が統計的に検証 | GrowthBook, Unleash |
| 9 | ガバナンス | モデルが規制・内部基準を満たす | MLflow Model Registry |
| 10 | 再学習 | モデル更新サイクルが自動化 | Airflow + MLflow |

### 学習優先度

| 優先度 | ツール |
|--------|--------|
| 最優先 | uv, pytest, Terraform, GitHub Actions, Docker, Kubernetes, Prometheus, CrossValidation |
| 高 | PytorchLightning, PyTorch, Hydra, MLflow, Airflow, ONNX |
| 中 | DVC, Great Expectations, ArgoCD, Evidently |
