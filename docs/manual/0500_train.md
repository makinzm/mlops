# 0500 — モデル学習

## 概要

前処理済みデータを入力として LightGBM で k-fold クロスバリデーション学習を行う。
学習結果（CV スコア・予測当たり外れ分析・feature importance）を `models/` 以下に保存する。

---

## 前提条件

- `uv run python -m src usecase=preprocess` が完了しており、
  `data/2026/Q1/processed/titanic_preprocess/` にタイムスタンプ付きディレクトリが存在すること。
- `lightgbm` がインストール済みであること（`uv sync` で自動インストールされる）。

---

## 設定の確認

### `conf/config.yaml`

```yaml
competition: titanic
```

### `conf/competition/titanic/training/lgbm.yaml`

| キー | 説明 |
|------|------|
| `preprocess_output_dir` | 前処理結果ディレクトリ。`latest` を含むと最新タイムスタンプに自動解決。 |
| `target_col` | 目的変数カラム名。 |
| `feature_cols` | 使用する特徴量カラムのリスト。 |
| `categorical_feature` | カテゴリ変数のリスト（LightGBM の category dtype で処理）。 |
| `sample_weight_col` | サンプル重みカラム名（null = 重みなし）。 |
| `loss.objective` | 損失関数（`binary` / `multiclass` 等）。 |
| `loss.metric` | 評価指標（`auc` / `binary_logloss` 等）。 |
| `lgbm.*` | LightGBM ハイパーパラメータ。 |
| `report.n_error_samples` | 当たり外れサンプリングの件数（カテゴリごと）。 |
| `output_dir` | モデル保存先。`models/${competition.name}` がデフォルト。 |
| `seed` | 乱数シード。 |

---

## 実行コマンド

### 全 trainer を実行（competition 配下の training/*.yaml を全て）

```bash
uv run python -m src usecase=train
```

### 特定の trainer のみ実行

```bash
uv run python -m src usecase=train trainer_name=lgbm
```

---

## 出力先

```
models/titanic/
└── titanic_lgbm/
    ├── .gitignore              # *.yaml と *.md のみ git 管理対象
    └── {YYYYMMDDTHHMMSS}/
        ├── train_result.yaml   # CV スコア・fold 詳細（git 管理）
        ├── README.md           # モデルレポート（git 管理）
        └── fold_0/
            ├── model.txt            # LightGBM モデル（git 除外）
            ├── oof_train.parquet    # OOF 予測値（git 除外）
            ├── error_analysis.parquet  # 当たり外れ分析（git 除外）
            └── feature_importance.parquet  # 特徴量重要度（git 除外）
```

---

## 出力ファイルの確認方法

### CV スコアの確認

```bash
cat models/titanic/titanic_lgbm/*/train_result.yaml
```

### README で fold スコアを確認

```bash
cat models/titanic/titanic_lgbm/*/README.md
```

### error_analysis の確認（Python）

```python
import pandas as pd

df = pd.read_parquet("models/titanic/titanic_lgbm/<timestamp>/fold_0/error_analysis.parquet")

# 最も自信を持って間違えた False Positive を確認
print(df[df["sample_type"] == "FP"].head(10))

# 見逃した生存者 (False Negative) を確認
print(df[df["sample_type"] == "FN"].head(10))
```

---

## サンプル重みの優先順位

```
sample_weight_col（行ごと）  >  class_weight（クラスごと）  >  is_unbalance（自動）
```

- `sample_weight_col: Weight` と指定すると、データの `Weight` カラムを重みとして使用する。
- `loss.class_weight: {0: 1.0, 1: 3.0}` のように指定すると、クラスごとに重みを設定する。
- `loss.is_unbalance: true` にすると LightGBM がクラス頻度に基づいて自動で重みを調整する。

---

## 実行ログの確認

```bash
ls outputs/         # Hydra が config スナップショットとログを保存
```
