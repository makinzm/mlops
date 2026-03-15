# 0600 — 推論・パイプライン実行

## 概要

学習済みモデルを使ってテストデータに対する予測を行い、Kaggle 提出用の `submission.csv` を生成する。
また、前処理 → 学習 → 推論の一連のステップを `usecase=pipeline` で一括実行できる。

---

## 前提条件

- `usecase=train`（または `usecase=train recipe=lgbm`）が完了しており、
  `models/titanic/titanic_lgbm/` 以下にタイムスタンプ付きディレクトリとモデルファイルが存在すること。
- `usecase=preprocess`（または `usecase=preprocess recipe=base`）が完了しており、
  `data/2026/Q1/processed/titanic_preprocess/` に test データが存在すること。

---

## 1. 推論（usecase=inference）

### 設定の確認

**`conf/competition/titanic/inference/titanic_ensemble.yaml`**

| キー | 説明 |
|------|------|
| `test_path` | テストデータのパス。`latest` で最新タイムスタンプを自動解決。 |
| `feature_cols` | 使用する特徴量カラムのリスト。 |
| `passenger_id_col` | 提出 CSV の ID カラム名。 |
| `models` | アンサンブル対象モデルのルートディレクトリリスト（各ディレクトリに `fold_N/` が必要）。 |
| `ensemble` | アンサンブル戦略（`mean` / `weighted_mean` / `rank_average`）。 |
| `weights` | `weighted_mean` 使用時の重みリスト（`models` と同じ数）。 |
| `output_dir` | 出力先ディレクトリ。 |

### 実行コマンド

```bash
# Titanic アンサンブル推論（titanic_ensemble.yaml を使用）
uv run python -m src usecase=inference recipe=titanic_ensemble
```

### 出力先

`data/2026/Q1/inference/titanic/titanic_inference/` 以下に以下のファイルが生成される：

```
titanic_inference/
├── .gitignore         # 自動生成（*.csv, *.yaml, *.md のみ保持）
├── submission.csv     # Kaggle 提出用 CSV（PassengerId, Survived）
├── metainfo.yaml      # 実行メタ情報（commit_hash, ensemble, n_models 等）
└── README.md          # 出力ファイルツリー
```

### アンサンブル戦略の選び方

| 戦略 | コマンドオプション | 説明 |
|------|------------------|------|
| 単純平均 | `ensemble: mean` | 複数モデルの予測を単純平均 |
| 重み付き平均 | `ensemble: weighted_mean` + `weights: [0.6, 0.4]` | 重みで加重平均（合計は自動正規化） |
| Rank Average | `ensemble: rank_average` | rank に変換してから平均（外れ値に robust） |

---

## 2. パイプライン（usecase=pipeline）

前処理 → 学習 → 推論のステップを定義順に自動実行する。

### 設定の確認

**`conf/competition/titanic/pipeline/all_after_download.yaml`**

```yaml
steps:
  - usecase: preprocess
    recipe: base
  - usecase: train
    recipe: lgbm
  - usecase: inference
    recipe: titanic_ensemble
```

各 step は `usecase` と `recipe` のペアで定義し、個別コマンドと同じ動作をする。

### 実行コマンド

```bash
# ダウンロード後の全ステップを一括実行
uv run python -m src usecase=pipeline recipe=all_after_download
```

### 失敗時の動作

- いずれかのステップが失敗した場合、後続ステップは実行されない（fail-fast）
- エラーメッセージを確認し、各 usecase を個別に実行してデバッグする

---

## 3. Target Encoding バリアント

Target Encoding 前処理を使ったパイプラインを実行する場合：

```bash
# Target Encoding で前処理
uv run python -m src usecase=preprocess recipe=target_encoding

# Target Encoding 済みデータで学習
uv run python -m src usecase=train recipe=lgbm_with_target_encoding
```

**注意**: Target Encoding は OOF（Out-of-Fold）CV で実装されているため、
Train データでのデータリークを防ぐ。Test データには Train の全体エンコーダーが適用される。

---

## 4. モデルパスの確認方法

`models/titanic/titanic_lgbm/` 以下のタイムスタンプディレクトリに fold ごとのモデルが保存される：

```
models/titanic/titanic_lgbm/
├── .gitignore
└── 20260315T120000/
    ├── train_result.yaml
    ├── README.md
    ├── fold_0/
    │   └── model.lgbm
    └── fold_1/
        └── model.lgbm
```

`titanic_ensemble.yaml` の `models` に `models/titanic/titanic_lgbm/latest` と設定すると、
最新タイムスタンプのモデルを自動選択する。

---

## 5. Kaggle への提出

生成された `submission.csv` を Kaggle に提出する：

1. `data/2026/Q1/inference/titanic/titanic_inference/submission.csv` を確認
2. Kaggle の Competition ページ → "Submit Predictions" から upload

提出ファイルの形式：

```csv
PassengerId,Survived
892,0
893,1
...
```
