# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MLOps project. The repository is currently in early development.

## Build and Development Commands

*To be added as the project develops.*

## Architecture

*To be documented as the codebase grows.*

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

### アーキテクチャ概要

```
Infrastructure → CI/CD → [実験 | データ | 検証] → デプロイ → [監視 | MLモニタリング | A/B] → [ガバナンス | 再学習]
```

### 学習優先度

| 優先度 | ツール |
|--------|--------|
| 最優先 | uv, pytest, Terraform, GitHub Actions, Docker, Kubernetes, Prometheus, CrossValidation |
| 高 | PytorchLightning, PyTorch, Hydra, MLflow, Airflow, ONNX |
| 中 | DVC, Great Expectations, ArgoCD, Evidently |

