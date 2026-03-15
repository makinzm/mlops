# 手動操作マニュアル: automatically_eda

## 概要

ダウンロードした CSV データセットを読み込み、EDA レポートを自動生成する。

出力先: `<report_dir>/<competition_name>_report/YYYYMMDD_HHMM/<analyzer_type>/`

---

## 前提条件

1. `uv sync` が完了していること
2. 分析対象の CSV が `data/` 以下に存在すること（先に download_dataset を実行）

---

## 基本実行

```bash
# Titanic データセットの EDA を実行（デフォルト設定）
uv run python -m src +competition/titanic/eda=eda
```

---

## 出力先

`analyze` に複数アナライザーを指定した場合、それぞれ独立したサブディレクトリに出力される。

```
competition/titanic_report/YYYYMMDD_HHMM/
├── pandas/
│   ├── .gitignore          # "*" で全ファイルを gitignore
│   ├── README.md           # ファイルごとの shape・欠損・スキーマ・実行ステップサマリー
│   ├── metainfo.yaml       # commit_hash, config, input_files, generated_at
│   ├── statistics/
│   │   ├── train_summary.parquet   (output_format=parquet の場合)
│   │   ├── train_summary.csv       (output_format=csv の場合)
│   │   └── train_missing.parquet / .csv
│   └── images/
│       ├── train_Age_dist.png
│       ├── train_Survived_dist.png
│       └── train_missing_heatmap.png
└── polars/                 (polars アナライザーを有効にした場合)
    ├── .gitignore
    ├── README.md
    ├── metainfo.yaml
    ├── statistics/
    └── images/
```

> **注意**: 各アナライザーディレクトリ内の `.gitignore` で管理され、git には追加されない。

---

## 設定のカスタマイズ

### `conf/usecase/automatically_eda.yaml` の構造

```yaml
# @package _global_
analyze:
  <analyzer_type>:         # pandas | polars
    output_format: parquet # pandas のみ有効: "parquet" | "csv"
                           # polars は常に parquet
    steps:
      - type: <step_type>  # 分析ステップ（後述）
      - type: <step_type>
        param_key: param_value
```

### input_paths の変更

```bash
# 特定のファイルを直接指定
uv run python -m src +competition/titanic/eda=eda \
  "input_paths=[data/2026/Q1/raw/titanic/train.csv]"

# 複数ファイルを指定
uv run python -m src +competition/titanic/eda=eda \
  "input_paths=[data/2026/Q1/raw/titanic/train.csv,data/2026/Q1/raw/titanic/test.csv]"
```

### 出力フォーマットの変更（pandas のみ）

```bash
# statistics を CSV で出力（デフォルトは parquet）
uv run python -m src +competition/titanic/eda=eda \
  "analyze.pandas.output_format=csv"
```

### アドホック分析の追加（yaml を直接編集）

`conf/usecase/automatically_eda.yaml` のコメントを外す:

```yaml
analyze:
  pandas:
    output_format: parquet
    steps:
      - type: basic_stats
      - type: distributions
      - type: missing_values
      # 以下のコメントを外す
      - type: group_stats
        group_by: "Survived"
      - type: id_transitions
        id_col: "PassengerId"
```

### Polars アナライザーを有効にする

```yaml
analyze:
  pandas:
    output_format: parquet
    steps:
      - type: basic_stats
      - type: distributions
  polars:
    steps:
      - type: distributions
```

---

## 分析ステップ一覧

| type | 説明 | 出力 |
|------|------|------|
| `basic_stats` | shape, dtypes, describe, 欠損値集計 | `statistics/<file>_summary.*`, `_missing.*` |
| `distributions` | 数値列: ヒストグラム / カテゴリ列: 棒グラフ | `images/<file>_<col>_dist.png` |
| `missing_values` | 欠損値ヒートマップ | `images/<file>_missing_heatmap.png` |
| `group_stats` | グループ集計（`group_by` 必須） | `statistics/<file>_group_stats.*`, `images/<file>_group_counts.png` |
| `id_transitions` | 重複 ID の遷移可視化（`id_col` 必須） | `images/<file>_id_transitions.png`（重複ありのみ） |

---

## アナライザーの違い

| アナライザー | 出力フォーマット | 備考 |
|-------------|----------------|------|
| `pandas` | `parquet` or `csv` | `output_format` で切替可能 |
| `polars` | `parquet` のみ | polars ネイティブ出力 |

---

## 新しいコンペティションを追加する方法

1. `conf/competition/<name>/competition.yaml` を作成:

```yaml
name: "house-prices"
```

2. `conf/competition/<name>/eda.yaml` を作成:

```yaml
# @package _global_
usecase: "automatically_eda"
seed: 42
report_dir: "competition"
max_plot_cols: 20
input_paths:
  - "data/2026/Q1/raw/house-prices"
analyze:
  pandas:
    output_format: parquet
    steps:
      - type: basic_stats
      - type: distributions
      - type: missing_values
```

3. 実行:

```bash
uv run python -m src +competition/house-prices/eda=eda
```

---

## 出力の確認方法

```bash
# レポートディレクトリを確認
ls competition/titanic_report/

# 最新レポートを確認
ls competition/titanic_report/$(ls competition/titanic_report/ | sort | tail -1)/

# pandas アナライザーの metainfo（commit hash など）を確認
cat competition/titanic_report/$(ls competition/titanic_report/ | sort | tail -1)/pandas/metainfo.yaml
```
