# 手動操作マニュアル: automatically_eda

## 概要

ダウンロードした CSV データセットを読み込み、EDA レポートを自動生成する。
出力先: `competition/<competition_name>_report/YYYYMMDD_HHMM/`

---

## 前提条件

1. `uv sync` が完了していること
2. 分析対象の CSV が `data/` 以下に存在すること（先に download_dataset を実行）

---

## 基本実行

```bash
# Titanic データセットの EDA を実行（デフォルト設定）
uv run python -m src usecase=automatically_eda competition=titanic
```

---

## 出力先

```
competition/titanic_report/YYYYMMDD_HHMM/
├── .gitignore          # "*" で全ファイルを gitignore
├── README.md           # ファイルごとの shape・欠損・実行ステップサマリー
├── metainfo.yaml       # commit_hash, config, input_files, generated_at
├── statistics/
│   ├── train_summary.parquet   (output_format=polars の場合)
│   └── train_missing.parquet
└── images/
    ├── train_Age_dist.png
    ├── train_Survived_dist.png
    └── train_missing_heatmap.png
```

> **注意**: `competition/` ディレクトリは各レポートの `.gitignore` で管理され、
> git には追加されない。

---

## 設定のカスタマイズ

### input_paths の変更

```bash
# 特定のファイルを直接指定
uv run python -m src usecase=automatically_eda competition=titanic \
  "competition.input_paths=[data/2026/Q1/raw/train.csv]"

# 複数ファイルを指定
uv run python -m src usecase=automatically_eda competition=titanic \
  "competition.input_paths=[data/2026/Q1/raw/train.csv,data/2026/Q1/raw/test.csv]"
```

### 出力フォーマットの変更

```bash
# statistics を CSV で出力（デフォルトは parquet）
uv run python -m src usecase=automatically_eda competition=titanic output_format=pandas
```

### アドホック分析の追加

```bash
# グループ統計を追加（Survived 列でグループ化）
uv run python -m src usecase=automatically_eda competition=titanic \
  "analyses=[{type:basic_stats},{type:distributions},{type:missing_values},{type:group_stats,group_by:Survived}]"

# ID 遷移可視化を追加（PassengerId 列で重複チェック）
uv run python -m src usecase=automatically_eda competition=titanic \
  "analyses=[{type:basic_stats},{type:id_transitions,id_col:PassengerId}]"
```

### conf/usecase/automatically_eda.yaml での永続設定

アドホック分析を常時有効にしたい場合は設定ファイルのコメントを外す:

```yaml
# conf/usecase/automatically_eda.yaml
analyses:
  - type: basic_stats
  - type: distributions
  - type: missing_values
  # 以下のコメントを外す
  - type: group_stats
    group_by: "Survived"
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

## 新しいコンペティションを追加する方法

1. `conf/competition/<name>.yaml` を作成:

```yaml
name: "house-prices"
input_paths:
  - "data/2026/Q1/raw"
```

2. 実行:

```bash
uv run python -m src usecase=automatically_eda competition=house-prices
```

---

## 出力の確認方法

```bash
# レポートディレクトリを確認
ls competition/titanic_report/

# 最新レポートを確認
ls competition/titanic_report/$(ls competition/titanic_report/ | sort | tail -1)/

# metainfo（commit hash など）を確認
cat competition/titanic_report/$(ls competition/titanic_report/ | sort | tail -1)/metainfo.yaml
```
