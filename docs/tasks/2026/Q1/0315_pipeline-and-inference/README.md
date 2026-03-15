# 実装計画（確定版）: Target Encoding / Inference / Pipeline Usecase

更新日: 2026-03-15

---

## 前提: 変更しないもの

- `conf/usecase/preprocess.yaml` — 変更なし（`usecase=preprocess` は引き続き動く）
- `conf/usecase/train.yaml` — 変更なし（`usecase=train` は引き続き動く）
- `conf/competition/titanic/preprocess/base.yaml` — 変更なし
- `conf/competition/titanic/training/lgbm.yaml` — 変更なし
- 既存テスト・既存 UseCase クラス — 変更なし

---

## 確定した実行コマンド

```bash
uv run python -m src usecase=preprocess recipe=target_encoding
uv run python -m src usecase=train recipe=lgbm_with_target_encoding
uv run python -m src usecase=inference recipe=titanic_ensemble
uv run python -m src usecase=pipeline recipe=all_after_download
```

---

## 追加するもの（3 件）

---

### ① Target Encoding（preprocess 拡張）

#### A. `SklearnResolver` に OOF Target Encoding を追加

`src/infrastructure/preprocessor/resolvers/sklearn_resolver.py` に追加するメソッド:

| メソッド | 役割 |
|---|---|
| `target_encode(df, columns, target_col, n_splits, smoothing, seed) -> (df, encoder)` | OOF CV で fold ごとに fit → OOF の行に encode。全 train で fit した encoder も返す |
| `transform_target_encode(df, encoder, columns) -> df` | fit 済み encoder を Test データに適用 |

**OOF CV 設計（リーク防止）:**

```python
# train 時
for fold_i, (train_idx, val_idx) in enumerate(kfold.split(df)):
    encoder_i = fit_encoder(df[train_idx], columns, target_col)
    df[val_idx, te_cols] = transform(df[val_idx], encoder_i)

# test 時
encoder_full = fit_encoder(train_df_full, columns, target_col)
test_df = transform(test_df, encoder_full)
```

**スムージング式:**
```
encoded_i = (n_i * mean_i + smoothing * global_mean) / (n_i + smoothing)
```
- `n_i`: カテゴリ i のサンプル数（fold の train 側）
- `mean_i`: カテゴリ i の target mean（fold の train 側）
- `global_mean`: fold の train 側全体の target mean
- `smoothing`: ハイパーパラメータ（デフォルト 1.0）

**未知カテゴリ:** `encoder["col"]["__global_mean__"]` でフォールバック（KeyError にしない）

**`execute()` 呼び出し時の動作:**
`fill_na` と同様に DAGRunner は `execute()` 経由で呼ぶため、OOF encode した結果の df を返す。
`execute()` は fit+transform 一体（返り値は変換済み df のみ）。
`target_encode()` / `transform_target_encode()` は直接呼び出し用として公開する。

#### B. 新規設定ファイル

**`conf/competition/titanic/preprocess/target_encoding.yaml`**
- `base.yaml` と同じ前処理 + `Sex`, `Embarked` の Target Encoding ステップを追加
- `job_id: titanic_te_preprocess`
- 全パスは Hydra Config 変数で管理

**`conf/competition/titanic/training/lgbm_with_target_encoding.yaml`**
- `lgbm.yaml` と同じ構造で以下を変更:
  - `job_id: titanic_lgbm_te`
  - `preprocess_output_dir`: target_encoding パイプラインの出力先（`latest` 使用）
  - `feature_cols`: `Sex_te`, `Embarked_te` を追加、`Sex`/`Embarked` の raw カラムを削除
  - `categorical_feature`: `Sex`, `Embarked` を削除（TE 後は数値カラムになるため）

#### C. `recipe=` パラメータの扱い

既存の `pipeline_loader.py` は `cfg.get("pipeline")` でファイルを絞り込む。
`recipe=` を新しいパラメータとして `pipeline_loader.py` / `trainer_loader.py` に追加する。

```bash
# recipe= を指定すると該当 yaml のみ実行
uv run python -m src usecase=preprocess recipe=target_encoding
# recipe= なしは従来通り全件実行
uv run python -m src usecase=preprocess
```

**変更方針:** `pipeline_loader.py` の `cfg.get("pipeline")` を `cfg.get("recipe", cfg.get("pipeline"))` に変更して後方互換を保つ。`trainer_loader.py` も同様に `cfg.get("trainer_name")` を `cfg.get("recipe", cfg.get("trainer_name"))` に変更。

---

### ② Inference Usecase 追加

#### 新規ファイル一覧

```
src/domain/model/inferencer.py              # Inferencer Protocol + InferenceResult
src/infrastructure/inferencer/
  __init__.py
  lgbm_inferencer.py                        # LightGBM fold モデルをロードして fold 平均予測
src/usecase/inference/
  __init__.py
  ensemble.py                               # EnsembleStrategy
  inference.py                              # InferenceUseCase
  inference_loader.py                       # load_inference_cfgs()

conf/usecase/inference.yaml                 # @package _global_ / usecase: inference
conf/competition/titanic/inference/
  titanic_ensemble.yaml

tests/usecase/inference/
  __init__.py
  test_ensemble.py
  test_inference_usecase.py
tests/infrastructure/inferencer/
  __init__.py
  test_lgbm_inferencer.py
```

#### Config 設計

```yaml
# conf/usecase/inference.yaml
# @package _global_
usecase: inference
# recipe: titanic_ensemble  # 省略すると competition/inference/ 全件実行
```

```yaml
# conf/competition/titanic/inference/titanic_ensemble.yaml
# @package _global_
usecase: inference
recipe: titanic_ensemble

ensemble:
  strategy: mean          # mean | weighted_mean | rank_average

models:
  - job_dir: ${competition.name}/titanic_lgbm   # models/ 配下の job_id ディレクトリ
    weight: 1.0

test_input:
  path: data/2026/Q1/processed/titanic_preprocess/latest/test_out.parquet
  id_col: PassengerId
  feature_cols:
    - Pclass
    - Age
    - SibSp
    - Parch
    - Fare
    - FamilySize
    - Sex
    - Embarked

output_dir: data/2026/Q1/submissions
seed: 42
```

#### `inference_loader.py` の設計

`load_trainer_cfgs()` / `load_pipeline_cfgs()` と同じ構造:
- `cfg.get("recipe")` が指定されていれば 1 件のみ
- 省略すると `competition/{name}/inference/*.yaml` を全件実行

#### Inferencer Protocol（domain 層）

```python
class Inferencer(Protocol):
    def predict_folds(
        self,
        model_ts_dir: Path,    # {job_dir}/{timestamp}/ — latest は UseCase 側で解決
        test_df: pl.DataFrame,
        feature_cols: list[str],
    ) -> np.ndarray: ...       # shape: (n_samples,)  fold 平均済み
```

#### InferenceResult（domain 層）

```python
@dataclass
class InferenceResult:
    recipe: str
    output_path: Path
    n_models: int
    strategy: str
    commit_hash: str
    seed: int
```

#### アンサンブル戦略

| strategy | 内容 |
|---|---|
| `mean` | 各モデルの予測値の単純平均 |
| `weighted_mean` | weights を正規化して加重平均 |
| `rank_average` | 各モデルの予測値を `rank / (n-1)` に変換してから平均 |

#### 出力構造

```
{output_dir}/{recipe}/{YYYYMMDD_HHMMSS}/
  submission.csv          # id_col, Survived（0/1 に二値化、threshold=0.5）
  inference_result.yaml   # recipe・strategy・models・commit_hash・seed
  README.md               # ファイルツリー構造
{output_dir}/
  .gitignore              # per-directory（*.csv 除外、*.yaml/*.md/*.gitignore/*.gitkeep/*/ 保持）
```

---

### ③ Pipeline Usecase 追加（逐次実行のみ）

#### 設計方針

「yaml に書いた steps を順番に実行する」だけ。ステージ間のデータ受け渡し・依存関係注入は不要。

`main.py` の各 elif ブロックを **関数として切り出す** ことでコード重複を防ぐ:

```python
def _run_preprocess(cfg: DictConfig, conf_dir: Path, logger) -> None: ...
def _run_train(cfg: DictConfig, conf_dir: Path, logger) -> None: ...
def _run_inference(cfg: DictConfig, conf_dir: Path, logger) -> None: ...
```

`PipelineUseCase.execute()` は steps を順番にループし、各 step の設定を base cfg にマージして対応する関数を呼ぶ。

#### 新規ファイル一覧

```
src/usecase/pipeline/
  __init__.py
  pipeline.py                       # PipelineUseCase + PipelineResult
  pipeline_loader.py                # load_pipeline_run_cfgs()（別名: usecase=pipeline 用）

conf/usecase/pipeline.yaml          # @package _global_ / usecase: pipeline
conf/competition/titanic/pipeline/
  all_after_download.yaml

tests/usecase/pipeline/
  __init__.py
  test_pipeline_usecase.py
```

#### Config 設計

```yaml
# conf/usecase/pipeline.yaml
# @package _global_
usecase: pipeline
# recipe: all_after_download  # 省略すると competition/pipeline/ 全件実行
```

```yaml
# conf/competition/titanic/pipeline/all_after_download.yaml
# @package _global_
usecase: pipeline
recipe: all_after_download

steps:
  - usecase: preprocess
    recipe: target_encoding
  - usecase: train
    recipe: lgbm_with_target_encoding
  - usecase: inference
    recipe: titanic_ensemble
```

#### PipelineResult

```python
@dataclass
class PipelineResult:
    recipe: str
    steps_executed: list[str]   # 実行した usecase 名リスト（順序保証）
    step_count: int
```

---

## ファイル変更サマリー

### 変更（既存ファイル）

| ファイル | 変更内容 |
|---|---|
| `src/infrastructure/preprocessor/resolvers/sklearn_resolver.py` | `target_encode` / `transform_target_encode` 追加、`supported_methods()` 更新 |
| `src/usecase/preprocessing/pipeline_loader.py` | `recipe=` パラメータ対応（`pipeline=` との後方互換あり） |
| `src/usecase/training/trainer_loader.py` | `recipe=` パラメータ対応（`trainer_name=` との後方互換あり） |
| `src/main.py` | `elif "inference"` / `elif "pipeline"` 追加。既存ブロックを関数化 |

### 新規追加（コード）

| ファイル | 概要 |
|---|---|
| `src/domain/model/inferencer.py` | `Inferencer` Protocol + `InferenceResult` dataclass |
| `src/infrastructure/inferencer/__init__.py` | 空 |
| `src/infrastructure/inferencer/lgbm_inferencer.py` | LightGBM fold 予測（fold 平均） |
| `src/usecase/inference/__init__.py` | 空 |
| `src/usecase/inference/ensemble.py` | `EnsembleStrategy`（mean / weighted_mean / rank_average） |
| `src/usecase/inference/inference.py` | `InferenceUseCase` |
| `src/usecase/inference/inference_loader.py` | `load_inference_cfgs()` |
| `src/usecase/pipeline/__init__.py` | 空 |
| `src/usecase/pipeline/pipeline.py` | `PipelineUseCase` + `PipelineResult` |
| `src/usecase/pipeline/pipeline_loader.py` | `load_pipeline_run_cfgs()` |

### 新規追加（設定）

| ファイル | 概要 |
|---|---|
| `conf/competition/titanic/preprocess/target_encoding.yaml` | OOF Target Encoding preprocess |
| `conf/competition/titanic/training/lgbm_with_target_encoding.yaml` | TE 特徴量使用 LightGBM |
| `conf/usecase/inference.yaml` | `usecase: inference` エントリポイント |
| `conf/competition/titanic/inference/titanic_ensemble.yaml` | アンサンブル設定 |
| `conf/usecase/pipeline.yaml` | `usecase: pipeline` エントリポイント |
| `conf/competition/titanic/pipeline/all_after_download.yaml` | steps 定義 |

### 新規追加（テスト）

| ファイル | 概要 |
|---|---|
| `tests/infrastructure/preprocessor/resolvers/test_sklearn_resolver.py` | `target_encode` テスト追加（既存ファイルに追記） |
| `tests/usecase/inference/__init__.py` | 空 |
| `tests/usecase/inference/test_ensemble.py` | EnsembleStrategy テスト |
| `tests/usecase/inference/test_inference_usecase.py` | InferenceUseCase E2E テスト |
| `tests/infrastructure/inferencer/__init__.py` | 空 |
| `tests/infrastructure/inferencer/test_lgbm_inferencer.py` | LightGBMInferencer テスト |
| `tests/usecase/pipeline/__init__.py` | 空 |
| `tests/usecase/pipeline/test_pipeline_usecase.py` | PipelineUseCase テスト |

---

## コミット計画（9 コミット）

| # | type | 内容 | --no-verify |
|---|---|---|---|
| 1 | `[chore]` | CI/lefthook で新テストが自動実行されることを確認 | なし |
| 2 | `[test] RED` | target_encode テスト実装 | yes |
| 3 | `[fix] GREEN` | SklearnResolver.target_encode + recipe 対応 + yaml 追加 | なし |
| 4 | `[test] RED` | inference テスト全件（ensemble / usecase / inferencer） | yes |
| 5 | `[fix] GREEN` | Inferencer / LightGBMInferencer / EnsembleStrategy / InferenceUseCase + conf | なし |
| 6 | `[test] RED` | pipeline テスト | yes |
| 7 | `[fix] GREEN` | PipelineUseCase + main.py 関数化 + conf | なし |
| 8 | `[refactor]` | 型アノテーション補完・コード整理 | なし |
| 9 | `[docs]` | `docs/manual/pipeline-and-inference.md` 作成 | なし |

---

## テストケース一覧

### ① Target Encoding（test_sklearn_resolver.py に追加）

| テスト名 | 検証内容 | 期待結果 |
|---|---|---|
| `test_target_encode_oof_no_leak` | OOF encode で val fold の行が fit 側に混入しないこと | val 行の encode に val 自身の統計量が使われていない |
| `test_target_encode_replaces_category_with_mean` | カテゴリが target 平均値に置換される | float カラムになる |
| `test_target_encode_unknown_category_uses_global_mean` | Test の未知カテゴリが `__global_mean__` でフォールバック | KeyError でなく数値が返る |
| `test_target_encode_smoothing_applied` | smoothing=0.0 vs smoothing=10.0 で値が変わる | 小カテゴリが global_mean 寄りになる |
| `test_target_encode_multiple_columns` | 複数カラムを同時にエンコードできる | 全カラムが変換される |
| `test_transform_target_encode_applies_train_encoder_to_test` | train encoder を test に適用できる | train 統計量が test に反映される |

### ② Inference

**test_ensemble.py:**

| テスト名 | 検証内容 | 期待結果 |
|---|---|---|
| `test_mean_strategy` | mean が単純平均を返す | 正しい平均値 |
| `test_weighted_mean_strategy` | weighted_mean が重みに応じた値を返す | weight=2:1 なら 2/3*A + 1/3*B |
| `test_rank_average_strategy` | rank_average が rank 変換後の平均を返す | rank 変換後の平均 |
| `test_weighted_mean_normalizes_weights` | weights が正規化される | [2.0, 2.0] → 等分 |

**test_inference_usecase.py:**

| テスト名 | 検証内容 | 期待結果 |
|---|---|---|
| `test_creates_submission_csv` | submission.csv が生成される | ファイルが存在する |
| `test_submission_csv_columns` | id_col + `Survived` の 2 列 | columns 一致 |
| `test_writes_result_yaml` | inference_result.yaml が生成される | ファイルが存在する |
| `test_result_yaml_has_commit_hash` | commit_hash が result.yaml に記録される | キーが存在する |
| `test_writes_gitignore` | output_dir に .gitignore が生成される | ファイルが存在する |
| `test_writes_readme` | README.md が生成される | ファイルが存在する |
| `test_raises_on_missing_model_dir` | 存在しないモデルディレクトリでエラー | `ValueError` |

**test_lgbm_inferencer.py:**

| テスト名 | 検証内容 | 期待結果 |
|---|---|---|
| `test_predict_folds_returns_array` | fold モデルをロードして shape=(n,) を返す | shape 一致 |
| `test_predict_folds_averages_folds` | 複数 fold の予測を fold 平均する | fold 平均が返る |

### ③ Pipeline

**test_pipeline_usecase.py:**

| テスト名 | 検証内容 | 期待結果 |
|---|---|---|
| `test_executes_steps_in_order` | steps が定義順に実行される | 実行順序が一致 |
| `test_result_contains_step_names` | PipelineResult に実行済み step 名が含まれる | `steps_executed` 一致 |
| `test_empty_steps_returns_zero_count` | steps が空のとき step_count=0 | `step_count == 0` |
| `test_raises_on_unknown_usecase_in_step` | 未知 usecase 名の step でエラー | `ValueError` |

---

## 完了条件

- [ ] `uv run pytest tests/infrastructure/preprocessor/ -v` 全通過（target_encode 含む）
- [ ] `uv run pytest tests/usecase/inference/ -v` 全通過
- [ ] `uv run pytest tests/infrastructure/inferencer/ -v` 全通過
- [ ] `uv run pytest tests/usecase/pipeline/ -v` 全通過
- [ ] `uv run pytest`（全テスト）通過
- [ ] lefthook pre-commit 通過
- [ ] 既存コマンド `usecase=preprocess` / `usecase=train`（recipe なし）が引き続き動作する
- [ ] 確定した実行コマンド 4 件が全て動作する
- [ ] per-directory .gitignore が submissions output_dir に動的生成される
- [ ] README.md が inference 出力ディレクトリに生成される
- [ ] inference_result.yaml に commit_hash が記録される
- [ ] seed で再現性が保たれる
- [ ] 全パスは Hydra Config で管理されている（ハードコードなし）
- [ ] `docs/manual/pipeline-and-inference.md` が存在する
