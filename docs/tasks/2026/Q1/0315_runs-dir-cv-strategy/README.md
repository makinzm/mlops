# runs/ ディレクトリ追加 + CV戦略の拡充

## 概要

### 変更1: preprocess_result.yaml を git 管理可能な runs/ に保存

**背景:**
現在 `preprocess_result.yaml` と `pipeline_dag.html` は `output_dir`（`data/processed/`）以下に保存されているが、
`data/**` は `.gitignore` 対象のため git に入らない。
メタデータ（実験のマニフェスト）は git 管理したい。

**変更内容:**
- `runs/` ディレクトリを新設（`.gitignore` に追加しない）
- `preprocess_result.yaml` と `pipeline_dag.html` を `runs/{job_id}/{timestamp}/` に保存する
- データファイル（parquet）は引き続き `output_dir/` に保存
- `conf/usecase/preprocess.yaml` に `runs_dir: runs` を追加
- `preprocess_result.yaml` に `output_dir` フィールドを追加してデータの場所を記録する

```
runs/                          # git-tracked（新設）
  {job_id}/
    {timestamp}/
      preprocess_result.yaml   # manifest（git管理）
      pipeline_dag.html        # DAG可視化（git管理）

data/processed/                # gitignored（変更なし）
  {job_id}/
    {timestamp}/
      tabular_out/             # データファイル
        fold_0/train.parquet
        ...
```

**変更ファイル:**
- `src/usecase/preprocessing/preprocess.py`: `runs_dir` を受け取り、そこに yaml と html を保存
- `conf/usecase/preprocess.yaml`: `runs_dir: runs` を追加
- `conf/usecase/preprocess_titanic.yaml`: `runs_dir: runs` を追加
- `runs/.gitkeep` を作成（ディレクトリを git に入れるため）

---

### 変更2: CV戦略の拡充

**背景:**
現在 `kfold` と `time_series` しか実装されていない。

**追加する CV 戦略:**

| strategy | 必要な追加パラメータ |
|----------|-------------------|
| `stratified_kfold` | `target_col` |
| `group_kfold` | `group_col` |
| `stratified_group_kfold` | `target_col`, `group_col` |
| `leave_one_group_out` | `group_col` |

**Config の変更（cv セクション）:**
```yaml
cv:
  strategy: stratified_kfold  # none/kfold/time_series/stratified_kfold/group_kfold/stratified_group_kfold/leave_one_group_out
  n_splits: 5
  time_col: null
  target_col: Survived         # stratified_kfold, stratified_group_kfold で使用
  group_col: null              # group_kfold, stratified_group_kfold, leave_one_group_out で使用
  input_id: null               # CVに使う入力ID（null=最初のinput）
```

**変更ファイル:**
- `src/usecase/preprocessing/preprocess.py`: `_build_cv_splits` に新戦略を追加
- `conf/usecase/preprocess.yaml`: CV 設定を更新
- `conf/usecase/preprocess_titanic.yaml`: `stratified_kfold` + `target_col: Survived` に更新

---

## 実装手順（TDD サイクル）

### サイクル1: runs/ ディレクトリ保存

- **RED**: `preprocess_result.yaml` と `pipeline_dag.html` が `runs_dir/` に保存されるテスト
- **GREEN**: `preprocess.py` に `runs_dir` 対応を追加
- commit & 報告

### サイクル2: CV戦略の拡充

- **RED**: `stratified_kfold`, `group_kfold`, `stratified_group_kfold`, `leave_one_group_out` のテスト
- **GREEN**: `_build_cv_splits` に新戦略を追加
- commit & 報告

### 最後の設定更新

- conf ファイルの更新（`preprocess.yaml`, `preprocess_titanic.yaml`）
- `runs/.gitkeep` の作成
- commit & 報告

---

## テストケース一覧

### サイクル1: runs/ ディレクトリ保存

1. `test_preprocess_result_yaml_saved_in_runs_dir`
   - 前提: `runs_dir` を指定した cfg
   - 期待: `runs/{job_id}/{timestamp}/preprocess_result.yaml` が存在する
   - 検証: `output_dir` には yaml が存在しない

2. `test_pipeline_dag_html_saved_in_runs_dir`
   - 前提: `runs_dir` を指定した cfg
   - 期待: `runs/{job_id}/{timestamp}/pipeline_dag.html` が存在する

3. `test_preprocess_result_yaml_contains_output_dir`
   - 前提: `runs_dir` を指定した cfg
   - 期待: `preprocess_result.yaml` に `output_dir` フィールドが含まれる

### サイクル2: CV戦略の拡充

1. `test_cv_stratified_kfold`
   - 前提: `strategy=stratified_kfold`, `target_col=label`, 10行データ（ラベル 0/1 各5行）
   - 期待: splits が 5 つ返る

2. `test_cv_group_kfold`
   - 前提: `strategy=group_kfold`, `group_col=group`, 10行データ（グループ 0-4 各2行）
   - 期待: splits が 5 つ返る

3. `test_cv_stratified_group_kfold`
   - 前提: `strategy=stratified_group_kfold`, `target_col=label`, `group_col=group`
   - 期待: splits が返る（n_splits <= グループ数）

4. `test_cv_leave_one_group_out`
   - 前提: `strategy=leave_one_group_out`, `group_col=group`, グループ 0-4 各2行
   - 期待: splits がグループ数（5）つ返る

5. `test_cv_input_id_selects_correct_df`
   - 前提: 複数 input, `input_id` で特定の df を指定
   - 期待: 指定した df を使って splits が生成される

---

## 完了条件

- [ ] サイクル1 RED → GREEN → REFACTOR 完了
- [ ] サイクル2 RED → GREEN → REFACTOR 完了
- [ ] `runs/.gitkeep` が git に入っている
- [ ] `conf/usecase/preprocess.yaml` と `preprocess_titanic.yaml` が更新されている
- [ ] 全テスト通過（`uv run pytest`）
- [ ] lefthook 通過
