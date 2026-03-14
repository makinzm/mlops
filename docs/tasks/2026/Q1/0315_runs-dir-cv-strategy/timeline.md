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

#### REFACTOR フェーズ

- `conf/usecase/preprocess.yaml`: `runs_dir: runs` と CV に `group_col`, `input_id` を追加
- `conf/usecase/preprocess_titanic.yaml`: `runs_dir: runs` 追加、`strategy: stratified_kfold` に変更
- `runs/.gitkeep` を作成

サイクル1 完了。

---

### サイクル2: CV 戦略の拡充

#### RED フェーズ開始

テスト対象:
- `test_cv_stratified_kfold`: strategy=stratified_kfold で 5 splits 返る
- `test_cv_group_kfold`: strategy=group_kfold で 5 splits 返る
- `test_cv_stratified_group_kfold`: strategy=stratified_group_kfold で splits 返る
- `test_cv_leave_one_group_out`: strategy=leave_one_group_out でグループ数の splits 返る
- `test_cv_input_id_selects_correct_df`: input_id で対象 df を指定できる

テストが失敗する理由:
- `_build_cv_splits` に stratified_kfold/group_kfold/stratified_group_kfold/leave_one_group_out が未実装

#### RED エラーログ

```
FAILED tests/usecase/preprocessing/test_preprocess.py::TestCvStrategy::test_cv_stratified_kfold
AssertionError: stratified_kfold で n_splits=5 の splits が生成されること
assert None == 5
```

`_build_cv_splits` が未知の strategy に対して None を返すため n_splits=None になる。RED フェーズ完了。

#### GREEN フェーズ

変更内容:
- `_build_cv_splits` に stratified_kfold/group_kfold/stratified_group_kfold/leave_one_group_out を追加
- `input_id` パラメータで CV に使う DataFrame を指定できるように対応

全 11 テスト通過。lefthook 通過。GREEN フェーズ完了。

#### REFACTOR フェーズ

大きなリファクタは不要。timeline 更新のみ。

サイクル2 完了。全タスク完了。
