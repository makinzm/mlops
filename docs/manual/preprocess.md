# 前処理パイプライン 手動確認手順

## 概要

`usecase=preprocess` を使用して、CSV/Parquet データに前処理変換を適用し
Parquet 形式で書き出す。DAG ベースのパイプラインで、Hydra Config により
変換内容・出力先・CV 戦略を宣言的に指定できる。

---

## 前提条件

```bash
uv sync
```

---

## 1. サンプルデータの準備

```bash
# テスト用 CSV を data/ 以下に作成する
uv run python -c "
import polars as pl
from pathlib import Path

Path('data/raw').mkdir(parents=True, exist_ok=True)
col1_vals = [None if i % 5 == 0 else float(i) for i in range(20)]
df = pl.DataFrame({
    'id': list(range(20)),
    'col1': col1_vals,
    'col2': [float(i * 2) for i in range(20)],
    'label': [i % 2 for i in range(20)],
})
df.write_csv('data/raw/sample_train.csv')
print('作成完了: data/raw/sample_train.csv')
print(df.head())
"
```

出力先: `data/raw/sample_train.csv`（`.gitignore` 対象）

---

## 2. 基本的な前処理の実行（cv=false）

```bash
uv run python -m src \
    usecase=preprocess \
    job_id=manual_test \
    "inputs=[{id: raw_train, path: data/raw/sample_train.csv, format: csv}]" \
    "steps=[{id: selected, polars: {method: select_columns, columns: [id, col1, label]}}, {id: filled, sklearn: {method: fill_na, strategy: median, columns: [col1]}}, {id: tabular_out, output: {columns: [id, col1, label], format: parquet, cv: false}}]" \
    "targets=[tabular_out]" \
    seed=42
```

### 確認ポイント

```bash
# 出力ディレクトリを確認
ls data/processed/manual_test/

# parquet が生成されていること
ls data/processed/manual_test/*/tabular_out.parquet

# 生成されたファイルの内容を確認
uv run python -c "
import polars as pl
import glob
files = glob.glob('data/processed/manual_test/**/tabular_out.parquet', recursive=True)
print(pl.read_parquet(files[0]))
"

# マニフェストを確認
cat data/processed/manual_test/*/preprocess_result.yaml

# DAG 可視化 HTML を確認（ブラウザで開く）
ls data/processed/manual_test/*/pipeline_dag.html
```

期待される出力ファイル:
```
data/processed/manual_test/{timestamp}/
  pipeline_dag.html
  preprocess_result.yaml
  tabular_out.parquet
```

---

## 3. Executor フォールバックの確認

未実装 Executor (`gcp_vertex`) を指定した場合に `local` にフォールバックすることを確認する。

```bash
uv run python -m src \
    usecase=preprocess \
    job_id=fallback_test \
    executor.type=gcp_vertex \
    "inputs=[{id: raw_train, path: data/raw/sample_train.csv, format: csv}]" \
    "steps=[{id: selected, polars: {method: select_columns, columns: [id, col1, label]}}, {id: tabular_out, output: {columns: [id, col1, label], format: parquet, cv: false}}]" \
    "targets=[tabular_out]"
```

### 確認ポイント

```bash
# preprocess_result.yaml に executor_fallback: true が記録されていること
cat data/processed/fallback_test/*/preprocess_result.yaml
```

期待される出力（抜粋）:
```yaml
executor_fallback: true
executor_requested: gcp_vertex
executor_used: local
```

---

## 4. CV 分割の確認（kfold）

```bash
uv run python -m src \
    usecase=preprocess \
    job_id=cv_test \
    "inputs=[{id: raw_train, path: data/raw/sample_train.csv, format: csv}]" \
    "cv={strategy: kfold, n_splits: 3}" \
    "steps=[{id: selected, polars: {method: select_columns, columns: [id, col1, label]}}, {id: tabular_out, output: {columns: [id, col1, label], format: parquet, cv: true}}]" \
    "targets=[tabular_out]"
```

### 確認ポイント

```bash
# fold_0, fold_1, fold_2 が生成されていること
ls data/processed/cv_test/*/tabular_out/

# fold_0 の train/test 行数を確認
uv run python -c "
import polars as pl, glob
train_files = sorted(glob.glob('data/processed/cv_test/**/fold_*/train.parquet', recursive=True))
test_files  = sorted(glob.glob('data/processed/cv_test/**/fold_*/test.parquet',  recursive=True))
for t, v in zip(train_files, test_files):
    print(f'train: {len(pl.read_parquet(t))} rows | test: {len(pl.read_parquet(v))} rows')
"
```

期待される出力ディレクトリ構造:
```
data/processed/cv_test/{timestamp}/tabular_out/
  fold_0/
    train.parquet
    test.parquet
  fold_1/
    train.parquet
    test.parquet
  fold_2/
    train.parquet
    test.parquet
```

---

## 5. graceful skip の確認（未実装 Resolver）

未実装の `torchvision` Resolver を含めた場合に、スキップされて後続ステップが継続することを確認する。

```bash
uv run python -m src \
    usecase=preprocess \
    job_id=skip_test \
    "inputs=[{id: raw_train, path: data/raw/sample_train.csv, format: csv}]" \
    "steps=[{id: embed_step, torchvision: {method: embed_image, column: col1}}, {id: selected, polars: {method: select_columns, columns: [id, col1, label]}}, {id: tabular_out, output: {columns: [id, col1, label], format: parquet, cv: false}}]" \
    "targets=[tabular_out]"
```

### 確認ポイント

```bash
# preprocess_result.yaml の step_results に status: skipped が含まれること
uv run python -c "
import yaml, glob
files = glob.glob('data/processed/skip_test/**/preprocess_result.yaml', recursive=True)
result = yaml.safe_load(open(files[0]))
for step in result['step_results']:
    print(step)
"
```

期待される出力（抜粋）:
```yaml
step_results:
  - resolver: torchvision
    method: embed_image
    status: skipped
    reason: resolver not found
  - resolver: polars
    method: select_columns
    status: ok
```

---

## 出力ファイルの説明

| ファイル | 説明 |
|---------|------|
| `pipeline_dag.html` | ブラウザで開くと DAG が表示される（Mermaid JS） |
| `preprocess_result.yaml` | 実行マニフェスト（学習コードの入口） |
| `{node_id}.parquet` | cv=false のときの単一出力 |
| `{node_id}/fold_N/train.parquet` | cv=true のときの fold 別 train データ |
| `{node_id}/fold_N/test.parquet` | cv=true のときの fold 別 test データ |

---

## テストの実行

```bash
# 全テスト
uv run pytest tests/ -v

# 前処理関連のみ
uv run pytest tests/domain/data/test_preprocessor.py \
              tests/infrastructure/preprocessor/ \
              tests/infrastructure/executor/ \
              tests/usecase/preprocessing/ -v
```
