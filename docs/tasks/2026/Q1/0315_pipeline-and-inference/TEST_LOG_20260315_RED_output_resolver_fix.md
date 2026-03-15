# TEST_LOG: RED phase — OutputResolver cv=False 出力形式修正

**日時**: 2026-03-15
**フェーズ**: RED (テスト更新→失敗確認)
**対象テストファイル**: `tests/infrastructure/preprocessor/resolvers/test_output_resolver.py`

## 背景・問題

`uv run python -m src usecase=inference` 実行時に以下のエラーが発生:

```
FileNotFoundError: data/2026/Q1/processed/titanic_preprocess/latest/test_out/test.parquet
```

## 根本原因

`OutputResolver.output()` の `cv=False` 時の出力先が `{node_id}.parquet`（フラットファイル）だった。
一方 `InferenceUseCase` は `{preprocess_output_dir}/test_out/test.parquet` を期待している。

```
現在の出力: {output_dir}/test_out.parquet        ← FileNotFoundError
期待する出力: {output_dir}/test_out/test.parquet  ← Inference が読む
```

## 実行コマンド

```bash
uv run pytest tests/infrastructure/preprocessor/resolvers/test_output_resolver.py -v
```

## 結果サマリー

```
2 failed, 2 passed
```

`TestOutputNoCV` の 2 件が失敗（期待通りの RED）。

## 失敗テスト

| テスト名 | エラー |
|---------|--------|
| `test_subdir_parquet_created` | `assert False` — `tabular_out/test.parquet` が存在しない |
| `test_parquet_content_matches` | `FileNotFoundError: tabular_out/test.parquet` |

## 修正方針

`OutputResolver.output()` の `cv=False` 分岐を変更:
- 変更前: `{output_dir}/{node_id}.parquet`（フラットファイル）
- 変更後: `{output_dir}/{node_id}/test.parquet`（サブディレクトリ）
