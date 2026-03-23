# リファクタ計画: Clean Architecture Naming 準拠

## 問題

mille の `name_deny` ルールにより、domain/usecase 層にインフラ固有の名前（`vertex`, `kaggle`, `gcp`, `lgbm`）を含められない。
これは単なる命名の問題ではなく、**責務の配置が間違っている**ことの表れ。

## 設計方針

### 原則
- Domain 層: インフラ非依存の抽象概念のみ
- UseCase 層: 「何をするか」のみ。「どこで・何を使って」は知らない
- Infrastructure 層: 具体実装。インフラ固有名はここだけに閉じる
- main.py: conf に基づいて DI

### 将来の拡張性
- リモート学習の「待機戦略」（同期/非同期）は `TrainingJobRepository` の Protocol で吸収
  - 今回: `run_custom_job`（同期: 送信 + 待機 + 結果返却）
  - 将来: `submit_job` + `get_job_status` を追加すれば UseCase を変えずに非同期対応可能

## 変更一覧

### Domain 層（リネーム: 抽象概念に）

| Before | After |
|--------|-------|
| `src/domain/repository/vertex_ai.py` | `src/domain/repository/training_job.py` |
| `VertexJobResult` | `TrainingJobResult` |
| `VertexAIRepository` | `TrainingJobRepository` |
| `src/domain/repository/gcs.py` | `src/domain/repository/object_storage.py` |
| `GCSRepository` | `ObjectStorageRepository` |

### UseCase 層（リネーム + コメント修正）

| Before | After |
|--------|-------|
| `src/usecase/training/vertex_train.py` | `src/usecase/training/remote_train.py` |
| `VertexAITrainUseCase` | `RemoteTrainUseCase` |
| `VertexTrainResult` | `RemoteTrainResult` |
| `src/usecase/kaggle_notebook/` | `src/usecase/notebook/` |
| `KaggleApiPort` | `NotebookPlatformPort` |
| `_staging.py` の `kaggleignore` | `ignore_patterns` |
| コメント内の vertex/kaggle/lgbm 参照 | 汎用的な表現に |

### Infrastructure 層（import パス変更のみ）

具体名はそのまま残す。Protocol の import 先だけ変更。

### main.py

import パス変更 + DI ロジック調整。

### テスト

import パス変更 + クラス名変更。

### conf

config ファイル名はインフラ固有名 OK（Hydra の usecase group 名として残す）。
`usecase:` の値を新しい名前に合わせる。
