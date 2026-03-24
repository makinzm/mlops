# Target Encoding マニュアル

## 概要

Target Encoding は、カテゴリカル変数をターゲット変数の統計量に基づいて数値に変換する手法。
本プロジェクトでは以下の 3 種類を提供する。

| 手法 | Resolver | 用途 |
|------|----------|------|
| OOF CV Target Encoding（smoothing ベース） | `SklearnResolver` | 一般的な TE。既存実装 |
| Bayesian Target Encoding（Beta-Binomial / Normal-Gamma） | `PolarsResolver` | 理論的根拠に基づく TE |
| 時系列 Expanding Window Target Encoding | `PolarsResolver` | 時系列データ向け TE |

## 1. suffix/prefix によるカラムリネーム（SklearnResolver）

### 設定変更箇所

パイプライン YAML の `target_encode` ノードに `suffix` / `prefix` を追加する。

```yaml
# conf/competition/<name>/pipeline/<pipeline>.yaml の該当ノード
- node_id: target_encode_sex
  resolver: sklearn
  method: target_encode
  params:
    columns: ["Sex", "Embarked"]
    target_col: "Survived"
    n_splits: 5
    seed: 42
    smoothing: 1.0
    suffix: "_te"       # Sex → Sex_te, Embarked → Embarked_te
```

### 実行コマンド

```bash
# 前処理パイプラインを実行（suffix 設定は YAML で指定済み）
uv run python -m src usecase=preprocess competition=titanic
```

### 出力カラム

- `suffix="_te"` 指定時: 元カラム `Sex` は保持されたまま、新カラム `Sex_te` が追加される
- 未指定（デフォルト）: 既存動作と同じく元カラムが float に上書きされる

## 2. Bayesian Target Encoding（PolarsResolver）

### 設定変更箇所

パイプライン YAML に `bayesian_target_encode` ノードを追加する。

```yaml
- node_id: bayesian_te
  resolver: polars
  method: bayesian_target_encode
  params:
    columns:
      - "Sex"                    # 単独カラム
      - ["Sex", "Embarked"]      # 複合キー（交互作用）
    target_col: "Survived"
    target_type: "binary"        # "binary" or "continuous"
    n_splits: 5
    seed: 42
    prior_weight: 1.0            # 大きいほど global_mean 寄り
    min_samples_leaf: 1          # この数未満は global_mean にフォールバック
    output_variance: true        # 事後分散カラムも出力
    suffix: "_bte"               # 出力カラム名接尾辞（デフォルト: "_te"）
```

### 実行コマンド

```bash
uv run python -m src usecase=preprocess competition=titanic
```

### 出力カラム

| 入力 | 出力（suffix="_bte"） | 説明 |
|------|----------------------|------|
| `Sex` | `Sex_bte` | Bayesian 事後平均 |
| `Sex` | `Sex_bte_var` | 事後分散（output_variance=true 時） |
| `["Sex", "Embarked"]` | `Sex_Embarked_bte` | 複合キー事後平均 |

### target_type の選び方

- `"binary"`: ターゲットが 0/1 の二値分類。Beta-Binomial モデルを使用
- `"continuous"`: ターゲットが連続値の回帰。Normal-Gamma モデルを使用

### prior_weight の調整指針

- `prior_weight=0.1`: データの観測値をほぼそのまま信頼する（過学習リスク高）
- `prior_weight=1.0`: デフォルト。バランスの取れた設定
- `prior_weight=10.0+`: global_mean に強く引き寄せる（保守的）

## 3. 時系列 Expanding Window Target Encoding（PolarsResolver）

### 設定変更箇所

```yaml
- node_id: time_series_te
  resolver: polars
  method: time_series_target_encode
  params:
    columns: ["Sex"]
    target_col: "Survived"
    time_col: "Date"             # 時系列カラム名（ソート用）
    target_type: "binary"
    prior_weight: 1.0
    min_samples: 1               # 履歴がこの数未満なら global_mean
    suffix: "_expanding"
```

### 実行コマンド

```bash
uv run python -m src usecase=preprocess competition=<name>
```

### 動作原理

1. `time_col` でデータを時系列順にソート
2. 各行 T について、T より前の行のみからカテゴリごとの Bayesian 統計量を計算
3. 履歴がない（最初の行など）場合は `global_mean` で埋める
4. 結果は元の行順序で返される

### 注意事項

- 時系列データでのみ使用すること。通常の KFold TE は未来の情報をリークする
- `transform` 時（Test データ）は全 Train データで fit した encoder が適用される

## テスト実行

```bash
# suffix/prefix テスト
uv run pytest tests/infrastructure/preprocessor/resolvers/test_sklearn_resolver.py -v -k "Suffix"

# Bayesian TE テスト
uv run pytest tests/infrastructure/preprocessor/resolvers/test_polars_resolver.py -v -k "Bayesian"

# 時系列 TE テスト
uv run pytest tests/infrastructure/preprocessor/resolvers/test_polars_resolver.py -v -k "TimeSeries"

# 全テスト
uv run pytest tests/ -v
```
