# TODO: 前処理パイプライン実装

## ステータス: 実装中

## 実装サイクル（RED -> GREEN の順序）

- [x] README.md 設計計画作成・承認
- [ ] **Cycle 1**: domain データクラス (PreprocessResult, StepResult, ColumnMeta, Node, DAG)
  - [ ] RED: テスト実装
  - [ ] GREEN: 実装
- [ ] **Cycle 2**: PolarsResolver (select_columns, arithmetic, exp_weight, join)
  - [ ] RED: テスト実装
  - [ ] GREEN: 実装
- [ ] **Cycle 3**: SklearnResolver (fill_na + データリーク防止)
  - [ ] RED: テスト実装
  - [ ] GREEN: 実装
- [ ] **Cycle 4**: OutputResolver + Registry + DAG Runner
  - [ ] RED: テスト実装
  - [ ] GREEN: 実装
- [ ] **Cycle 5**: Executor (LocalExecutor + Factory) + PreprocessUseCase + main.py + conf
  - [ ] RED: テスト実装
  - [ ] GREEN: 実装
- [ ] **Cycle 6**: Refactor
- [ ] **Cycle 7**: docs/manual/preprocess.md 作成
- [ ] PR 作成

## ファイル一覧（新規作成）

### src/
- `src/domain/data/preprocessor.py` — PreprocessResult, StepResult, ColumnMeta, Node, DAG
- `src/domain/executor/executor.py` — Executor Protocol
- `src/infrastructure/preprocessor/__init__.py`
- `src/infrastructure/preprocessor/registry.py` — RESOLVER_REGISTRY
- `src/infrastructure/preprocessor/resolvers/__init__.py`
- `src/infrastructure/preprocessor/resolvers/base.py` — StepResolver Protocol
- `src/infrastructure/preprocessor/resolvers/polars_resolver.py`
- `src/infrastructure/preprocessor/resolvers/sklearn_resolver.py`
- `src/infrastructure/preprocessor/resolvers/output_resolver.py`
- `src/infrastructure/preprocessor/dag_runner.py`
- `src/infrastructure/preprocessor/visualizer.py`
- `src/infrastructure/executor/__init__.py`
- `src/infrastructure/executor/local.py`
- `src/infrastructure/executor/factory.py`
- `src/usecase/preprocessing/__init__.py`
- `src/usecase/preprocessing/preprocess.py`

### conf/
- `conf/usecase/preprocess.yaml`
- `conf/executor/local.yaml`
- `conf/executor/ray_local.yaml`
- `conf/executor/gcp_vertex.yaml`

### tests/
- `tests/domain/data/test_preprocessor.py`
- `tests/infrastructure/preprocessor/resolvers/test_polars_resolver.py`
- `tests/infrastructure/preprocessor/resolvers/test_sklearn_resolver.py`
- `tests/infrastructure/preprocessor/resolvers/test_output_resolver.py`
- `tests/infrastructure/preprocessor/test_registry.py`
- `tests/infrastructure/preprocessor/test_dag_runner.py`
- `tests/infrastructure/executor/test_factory.py`
- `tests/usecase/preprocessing/test_preprocess.py`

### docs/
- `docs/manual/preprocess.md`
