# 0100: コンペティション conf のセットアップ

新しい Kaggle コンペを始めるとき、最初に conf を用意する。
conf は Hydra の config group として管理し、コンペ名でグループを分ける。

---

## conf 全体構造

```
conf/
├── config.yaml                    # ルート: デフォルト競技と _self_ のみ
├── competition/
│   └── {name}/
│       ├── competition.yaml       # コンペ識別子（name のみ）
│       ├── eda.yaml               # EDA: input/output パスと分析ステップ
│       └── preprocess/
│           ├── base.yaml          # 共通前処理パイプライン
│           ├── lgbm.yaml          # LightGBM 用（アンサンブル時）
│           └── nn.yaml            # NN 用（アンサンブル時）
├── usecase/
│   ├── automatically_eda.yaml     # EDA usecase のデフォルト設定
│   └── download_dataset.yaml      # ダウンロード usecase の設定
└── executor/
    ├── local.yaml
    ├── ray_local.yaml
    └── gcp_vertex.yaml
```

---

## 新しいコンペを始めるとき

### 1. competition.yaml を作成

```bash
mkdir -p conf/competition/{name}/preprocess
```

```yaml
# conf/competition/{name}/competition.yaml
name: "{name}"
```

例（house-prices の場合）:
```yaml
# conf/competition/house-prices/competition.yaml
name: "house-prices"
```

### 2. eda.yaml を作成

```yaml
# conf/competition/{name}/eda.yaml
# @package _global_
usecase: "automatically_eda"
seed: 42
output_dir: "reports"
max_plot_cols: 20

input_paths:
  - "data/2026/Q1/raw/{name}"

analyze:
  pandas:
    output_format: parquet
    steps:
      - type: basic_stats
      - type: distributions
      - type: missing_values
      # - type: group_stats
      #   group_by: "Target"
      # - type: id_transitions
      #   id_col: "Id"
```

> **注意**: `input_paths` と `output_dir` は異なるディレクトリにすること。
> 同一または包含関係にある場合は実行時に `ValueError` で検出される。

### 3. preprocess/base.yaml を作成

```yaml
# conf/competition/{name}/preprocess/base.yaml
# @package _global_
usecase: preprocess
job_id: {name}_preprocess

inputs:
  - id: raw_train
    path: data/2026/Q1/raw/{name}/train.csv
    format: csv
  - id: raw_test
    path: data/2026/Q1/raw/{name}/test.csv
    format: csv

output_dir: data/processed/{name}

cv:
  strategy: stratified_kfold
  n_splits: 5
  target_col: Target
  seed: 42

pipeline:
  - id: select
    from: raw_train
    method: "polars:select_columns"
    params:
      columns: [Id, Feature1, Feature2, Target]
  - id: fill
    from: select
    method: "sklearn:fill_na"
    params:
      strategy: median
  - id: output
    from: [fill]
    method: "output:save_parquet"
    params:
      cv: true
```

### 4. config.yaml のデフォルト競技を変更

```yaml
# conf/config.yaml
defaults:
  - competition: {name}/competition
  - _self_
```

---

## コンペ切り替え（デフォルト変更なし）

config.yaml を変更せず CLI で一時的に別コンペを使うことも可能。

```bash
# デフォルト以外のコンペでダウンロード
uv run python -m src +usecase=download_dataset \
  'competition=house-prices/competition'

# デフォルト以外のコンペで EDA
uv run python -m src +competition/house-prices=eda
```

---

## 実行コマンド早見表

| 操作 | コマンド |
|------|---------|
| ダウンロード（デフォルト競技） | `uv run python -m src +usecase=download_dataset` |
| EDA（デフォルト競技） | `uv run python -m src competition/titanic=eda` |
| 前処理（base） | `uv run python -m src '+competition/titanic/preprocess=base'` |

---

## アーカイブ（コンペ終了後）

```bash
mkdir -p conf/archive/2026/Q1/
git mv conf/competition/titanic conf/archive/2026/Q1/titanic
git commit -m "archive: titanic"
```

その後 `conf/config.yaml` の `competition:` を次のコンペに切り替える。
