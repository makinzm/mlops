# 前処理パイプライン

## 実行

```bash
uv run python -m src usecase=competition/titanic/preprocess/base
```

---

## 設定ファイルの場所

コンペごとの前処理設定は `conf/competition/{name}/preprocess/` に置く。

```
conf/competition/titanic/preprocess/
  base.yaml      # 共通前処理
  lgbm.yaml      # LightGBM 用（アンサンブル時）
  nn.yaml        # NN 用（アンサンブル時）
```

新しい設定を追加したい場合は `conf/usecase/preprocess.yaml` をテンプレートとしてコピーして編集する。

```bash
cp conf/usecase/preprocess.yaml conf/competition/titanic/preprocess/lgbm.yaml
```

---

## 設定のポイント

**入力データ**
```yaml
inputs:
  - id: raw_train
    path: data/2026/Q1/raw/train.csv
    format: csv   # csv / parquet
```

**CV 戦略**
```yaml
cv:
  strategy: stratified_kfold   # none / kfold / stratified_kfold / group_kfold / ...
  n_splits: 5
  target_col: Survived         # stratified_kfold で必要
  group_col: null              # group_kfold で必要
```

**変換ステップ（DAG）**
```yaml
steps:
  - id: selected
    polars:
      method: select_columns
      columns: [PassengerId, Survived, Age, Fare]

  - id: filled
    sklearn:
      method: fill_na
      strategy: median
      columns: [Age, Fare]

  - id: train_out
    output:
      columns: [PassengerId, Survived, Age, Fare]
      format: parquet
      cv: true      # true → fold_N/train.parquet + test.parquet に分割

targets: [train_out]
```

利用できる resolver / method の一覧:

| resolver | method | 主なパラメータ |
|----------|--------|--------------|
| `polars` | `select_columns` | `columns` |
| `polars` | `arithmetic` | `operation`, `col_a`, `col_b`, `output_col` |
| `polars` | `exp_weight` | `time_col`, `decay`, `weight_col` |
| `polars` | `join` | `on`, `how` |
| `sklearn` | `fill_na` | `strategy`(median/mean/constant), `columns` |

---

## 出力

```
data/processed/{job_id}/{timestamp}/
  preprocess_result.yaml   # 実行マニフェスト（git 管理対象）
  pipeline_dag.html        # DAG 可視化（ブラウザで開く）
  {output_node_id}.parquet             # cv: false
  {output_node_id}/fold_N/train.parquet  # cv: true
  {output_node_id}/fold_N/test.parquet
```

---

## コンペ設定のアーカイブ

→ [EOL_conf.md](./EOL_conf.md) を参照
