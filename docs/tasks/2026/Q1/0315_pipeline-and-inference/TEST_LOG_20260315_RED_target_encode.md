# TEST_LOG: RED phase — target_encode テスト

**日時**: 2026-03-15
**フェーズ**: RED (テスト実装→失敗確認)
**対象テストファイル**: `tests/infrastructure/preprocessor/resolvers/test_sklearn_resolver.py`

## 実行コマンド

```bash
uv run pytest tests/infrastructure/preprocessor/resolvers/test_sklearn_resolver.py -v
```

## 結果サマリー

```
7 failed, 7 passed in 0.95s
```

- **既存テスト（TestFillNaMedian）**: 7 passed（全通過）
- **新規テスト（TestTargetEncode）**: 7 failed（全失敗）— 期待通りの RED 状態

## 失敗したテスト一覧

| テスト名 | エラー |
|---------|--------|
| `test_target_encode_oof_no_leak` | `AttributeError: 'SklearnResolver' object has no attribute 'target_encode'` |
| `test_target_encode_replaces_category_with_mean` | `AttributeError: 'SklearnResolver' object has no attribute 'target_encode'` |
| `test_target_encode_unknown_category_uses_global_mean` | `AttributeError: 'SklearnResolver' object has no attribute 'target_encode'` |
| `test_target_encode_smoothing_applied` | `AttributeError: 'SklearnResolver' object has no attribute 'target_encode'` |
| `test_target_encode_multiple_columns` | `AttributeError: 'SklearnResolver' object has no attribute 'target_encode'` |
| `test_transform_target_encode_applies_train_encoder_to_test` | `AttributeError: 'SklearnResolver' object has no attribute 'target_encode'` |
| `test_supported_methods_includes_target_encode` | `AssertionError: assert 'target_encode' in {'fill_na'}` |

## エラー詳細（代表例）

```
AttributeError: 'SklearnResolver' object has no attribute 'target_encode'
```

`SklearnResolver` に `target_encode` / `transform_target_encode` が未実装のため、
全テストが AttributeError または AssertionError で失敗している。これは期待通りの RED 状態。

## 次のステップ

GREEN フェーズ: `SklearnResolver` に以下を実装する。

1. `target_encode(df, columns, target_col, n_splits, seed, smoothing=1.0)` — OOF CV ベースの TE
2. `transform_target_encode(df, encoder, columns)` — full_encoder を test に適用
3. `supported_methods()` に `"target_encode"` を追加
