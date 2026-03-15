# TEST_LOG — RED Phase: test_path=null support

Date: 2026-03-15

## 何をしようとしていたか

InferenceUseCase が `test_path=null` またはファイル不在のとき
submission.csv をスキップして metainfo.yaml / README.md のみ生成するロジックを追加する。
その前提として RED テストを先に書いてエラーを確認した。

## テストケース

1. `test_run_skips_submission_when_no_test_path`
2. `test_run_skips_submission_when_test_file_missing`

## エラーログ（RED確認）

```
FAILED tests/usecase/inference/test_inference_usecase.py::TestInferenceUseCaseRun::test_run_skips_submission_when_no_test_path
  - FileNotFoundError: No such file or directory (os error 2): None

FAILED tests/usecase/inference/test_inference_usecase.py::TestInferenceUseCaseRun::test_run_skips_submission_when_test_file_missing
  - FileNotFoundError: No such file or directory (os error 2): .../processed/test.parquet
```

現在の `InferenceUseCase.run()` は `test_path` の存在チェックをしていないため、
`pl.read_parquet()` が `FileNotFoundError` を投げている。

## 次のステップ（GREEN）

`InferenceUseCase.run()` の先頭で以下を確認し、
条件を満たさない場合は submission.csv 生成をスキップする:

1. `cfg.test_path is None` or `cfg.test_path == "null"`
2. `Path(cfg.test_path).exists() is False`

スキップ時も metainfo.yaml / README.md / .gitignore は生成する。
