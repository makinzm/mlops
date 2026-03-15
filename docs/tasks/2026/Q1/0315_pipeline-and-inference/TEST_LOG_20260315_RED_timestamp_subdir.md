# TEST_LOG — RED Phase: timestamp subdir for InferenceUseCase

Date: 2026-03-15

## 何をしようとしていたか

InferenceUseCase の出力ディレクトリを train/preprocess と同様に
`{output_dir}/{job_id}/{YYYYMMDDTHHMMSS}/` 形式に変更する。
その前提として RED テストを先に書いてエラーを確認した。

## テストケース

1. `test_run_output_dir_has_timestamp_subdir`
2. `test_run_returns_submission_path_in_timestamp_dir`
3. `test_run_metainfo_records_timestamp`

## エラーログ（RED確認）

```
FAILED ...::test_run_output_dir_has_timestamp_subdir
  AssertionError: job_id ディレクトリが生成されていない: .../inference_out/titanic_inference

FAILED ...::test_run_returns_submission_path_in_timestamp_dir
  AssertionError: timestamp ディレクトリ名が YYYYMMDDTHHMMSS 形式でない: inference_out
  assert 13 == 15

FAILED ...::test_run_metainfo_records_timestamp
  AssertionError: metainfo.yaml に timestamp が記録されていない
  assert 'timestamp' in {'job_id': ..., 'commit_hash': ..., 'ensemble': ..., ...}
```

現在の `InferenceUseCase.run()` は `output_dir` 直下にファイルを生成しており、
`{job_id}/{timestamp}/` サブディレクトリを作成していない。

## 次のステップ（GREEN）

`InferenceUseCase.run()` を以下のように変更する:

1. `job_output_dir = output_dir / job_id` を作成
2. `.gitignore` を `job_output_dir` に配置（train と同じ）
3. `timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")` を生成
4. `ts_dir = job_output_dir / timestamp` を作成し、全ファイルをここに配置
5. metainfo.yaml に `timestamp` フィールドを追加
6. 既存テストも `ts_dir` ベースで探すよう更新が必要
