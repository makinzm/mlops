# Target Encoding Infrastructure 改善

## Context

現在の Target Encoding は `Sex` → `Sex`（float）のようにカラム名が変わらず、後から読む人が混乱する。
また、Bayesian Target Encoding（Beta-Binomial / Normal-Gamma）と時系列対応の Expanding Window TE が未実装。
Titanic 固有の config 変更は行わず、インフラ層の汎用機能として実装する。

## 変更対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| `src/infrastructure/preprocessor/resolvers/polars_resolver.py` | Bayesian TE、時系列 TE を Polars ネイティブで新規実装 |
| `src/infrastructure/preprocessor/resolvers/sklearn_resolver.py` | 既存 `target_encode` / `transform_target_encode` に suffix/prefix を追加 |
| `tests/infrastructure/preprocessor/resolvers/test_polars_resolver.py` | Polars TE の全テスト追加 |
| `tests/infrastructure/preprocessor/resolvers/test_sklearn_resolver.py` | suffix/prefix テスト追加 |

## 機能一覧

### 機能 1: suffix/prefix によるカラムリネーム（sklearn_resolver）

- `suffix="_te"` → `Sex` → `Sex_te`
- デフォルト空文字で後方互換
- encoder dict のキーは元のカラム名のまま
- suffix 指定時、元カラムは保持する（drop しない）

### 機能 2: Bayesian Target Encoding（polars_resolver）

- 二値分類: Beta-Binomial 事後平均
- 連続値: Normal-Gamma 事後平均
- 複合キーグルーピング: `["Sex", "Embarked"]` → `Sex_Embarked_te`
- OOF CV でデータリーク防止
- 事後分散出力オプション

### 機能 3: 時系列 Target Encoding — Expanding Window（polars_resolver）

- `time_col` でソート → 各行で自分より前の行のみから Bayesian 統計量を計算
- 履歴がない行は事前分布の平均で埋める
- 元の行順序を保持

## テスト一覧

### sklearn_resolver: TestTargetEncodeSuffix (6 tests)
- `test_suffix_renames_columns`
- `test_prefix_renames_columns`
- `test_suffix_and_prefix_combined`
- `test_no_suffix_backward_compatible`
- `test_transform_suffix_applied`
- `test_encoder_keys_unchanged`

### polars_resolver: TestBayesianTargetEncode (10 tests)
- `test_binary_basic`
- `test_binary_posterior_mean_correctness`
- `test_binary_posterior_variance_output`
- `test_binary_prior_weight_effect`
- `test_binary_min_samples_leaf`
- `test_binary_oof_no_leak`
- `test_continuous_basic`
- `test_continuous_posterior_mean_correctness`
- `test_unknown_category_fallback`
- `test_supported_method`

### polars_resolver: TestBayesianTargetEncodeGrouping (4 tests)
- `test_composite_key_grouping`
- `test_composite_key_output_column_name`
- `test_mixed_single_and_composite`
- `test_composite_key_unknown_combination`

### polars_resolver: TestTimeSeriesTargetEncode (8 tests)
- `test_basic`
- `test_no_future_leak`
- `test_first_row_uses_prior`
- `test_expanding_mean_correctness`
- `test_min_samples`
- `test_with_suffix`
- `test_preserves_row_order`
- `test_transform_time_series_applies_full_encoder`

## 実装順序

1. ブランチ作成 + 計画記録
2. lefthook / CI 確認
3. RED: 全テスト実装
4. GREEN: 実装
5. REFACTOR: docstring 整備
6. マニュアル作成
7. PR 作成
