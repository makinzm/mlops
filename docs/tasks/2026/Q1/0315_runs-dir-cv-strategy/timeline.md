# Timeline

## 2026-03-15

### サイクル1: runs/ ディレクトリ保存

#### RED フェーズ開始

テスト対象:
- `test_preprocess_result_yaml_saved_in_runs_dir`
- `test_pipeline_dag_html_saved_in_runs_dir`
- `test_preprocess_result_yaml_contains_output_dir`

テストが失敗する理由:
- `preprocess.py` が `runs_dir` パラメータを受け取らない
- yaml/html は現在 `output_dir/` に保存されており `runs_dir/` には保存されない

#### RED エラーログ

```
FAILED tests/usecase/preprocessing/test_preprocess.py::TestRunsDir::test_preprocess_result_yaml_saved_in_runs_dir
AssertionError: preprocess_result.yaml が runs_dir/ 以下に存在すること
assert 0 >= 1
```

3テストとも同様に失敗（runs_dir/ にファイルが生成されない）。RED フェーズ完了。

#### GREEN フェーズ

変更内容:
- `preprocess.py`: `runs_dir` パラメータを受け取り、`manifest_dir` を分離
- `manifest_dir = runs_dir/job_id/timestamp`（runs_dir 指定時）
- `_build_manifest` に `output_dir` フィールドを追加

全 6 テスト通過。lefthook 通過。GREEN フェーズ完了。
