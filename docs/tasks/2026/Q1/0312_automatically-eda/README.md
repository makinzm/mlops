# Task: automatically_eda usecase

## 概要

ダウンロードしたデータセットをどう前処理すればよいかわからない問題に対し、
EDA レポートを自動生成することで前処理の起点を提供する。

CSV データを読み込み、以下を生成して `competition/<title>_report/YYYYMMDD_HHMM/` に出力する:
- 基本統計（shape, dtypes, describe, 欠損値集計）
- 分布プロット（ヒストグラム・棒グラフ）
- 欠損値ヒートマップ
- グループ統計（オプション）
- ID 遷移可視化（オプション）

## 設計

### アーキテクチャ

Clean Architecture に従い以下のレイヤーで実装する:

```
CLI (main.py)
  → AutomaticallyEDAUseCase
    → DataAnalyzer Protocol (domain)
      ← PandasAnalyzer (infrastructure)
```

### config 構成

```
conf/
├── config.yaml               ← defaults に competition: titanic 追加
├── competition/
│   └── titanic.yaml          ← name, input_paths
└── usecase/
    └── automatically_eda.yaml ← analyses list, output_format, seed, report_dir
```

### 分析ステップ

| type | 汎用/アドホック | 出力 |
|------|---------------|------|
| `basic_stats` | 汎用 | statistics/<file>_summary.*, _missing.* |
| `distributions` | 汎用 | images/<file>_<col>_dist.png |
| `missing_values` | 汎用 | images/<file>_missing_heatmap.png |
| `group_stats` | アドホック | statistics/<file>_group_stats.*, images/<file>_group_counts.png |
| `id_transitions` | アドホック | images/<file>_id_transitions.png（重複IDあり時のみ） |

### 出力形式

| output_format | statistics | images |
|--------------|-----------|--------|
| `polars` | .parquet | .png |
| `pandas` | .csv | .png |

## 実装順序

1. [x] feature branch 作成
2. [x] タスク docs 作成
3. [ ] pyproject.toml: matplotlib>=3.7, polars>=0.20 追加
4. [ ] RED: tests 全件失敗確認 → --no-verify commit → TEST_LOG 保存
5. [ ] GREEN: domain → usecase → infrastructure → config → main.py
6. [ ] CI チェック (pytest + m-y-p-y + ruff)
7. [ ] マニュアル: docs/manual/automatically-eda.md
8. [ ] PR 作成

## 変更・追加ファイル

### 新規

- `conf/competition/titanic.yaml`
- `conf/usecase/automatically_eda.yaml`
- `src/domain/data/eda.py`
- `src/usecase/eda/__init__.py`
- `src/usecase/eda/automatically_eda.py`
- `src/infrastructure/analyzer/__init__.py`
- `src/infrastructure/analyzer/pandas_analyzer.py`
- `tests/domain/data/test_eda.py`
- `tests/usecase/eda/__init__.py`
- `tests/usecase/eda/test_automatically_eda.py`
- `tests/infrastructure/analyzer/__init__.py`
- `tests/infrastructure/analyzer/test_pandas_analyzer.py`
- `docs/manual/automatically-eda.md`

### 修正

- `conf/config.yaml` — defaults に `competition: titanic` 追加
- `conf/usecase/download_dataset.yaml` — `usecase: download_dataset` 追加
- `src/main.py` — `_resolve_analyzer()` + usecase ルーティング追加
- `pyproject.toml` — matplotlib, polars 追加

## 検証方法

```bash
# テスト
uv run pytest && uv run m-y-p-y tests/ src/ && uv run ruff check .

# 手動確認
uv run python -m src usecase=automatically_eda competition=titanic
# → competition/titanic_report/YYYYMMDD_HHMM/ 生成確認
```
