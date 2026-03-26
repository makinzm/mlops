# Refactor: main.py の分割

## 背景

`src/main.py` が 597 行に肥大化している。主な問題:

1. **Kaggle 認証ボイラープレートが 3 箇所で重複** (`push_notebook`, `update_source_dataset_pipeline`, `create/update_source_dataset`)
2. **`main()` 関数内の巨大な if/elif チェーン** (13 分岐)
3. **ヘルパー関数とランナー関数が 1 ファイルに混在** (`_ensure_cloud_config`, `_resolve_manifest_path`, `_run_*` 等)

## ゴール

- `main.py` を 100 行以下に削減
- 関心ごとに分離された presentation 層モジュールを作成
- 既存テスト (`tests/test_main_resolve.py`) の import を更新
- `PipelineUseCase` への DI インターフェースは変更なし

## 設計

Clean Architecture に従い `src/presentation/` ディレクトリを新設する:

```
src/presentation/
├── __init__.py
├── kaggle_auth.py       # Kaggle API 認証ヘルパー（重複排除）
├── cloud_config.py      # _ensure_cloud_config, _load_trainer_cfgs_safe, _resolve_manifest_path
├── runners.py           # 全 _run_* 関数 + _resolve_downloader, _resolve_analyzers 等
├── registry.py          # usecase 名 → runner 関数のレジストリ + dispatch
```

`src/main.py` は以下だけになる:
- `load_dotenv`, logging 設定, `_CONF_DIR`
- `@hydra.main` デコレータ
- `platform_username` 注入
- `registry.dispatch(usecase_name, cfg, logger)` 呼び出し

### 移動先の対応表

| 現在の場所 (main.py) | 移動先 |
|---|---|
| `_resolve_downloader()` | `presentation/runners.py` |
| `_parse_analyses()`, `_resolve_analyzers()` | `presentation/runners.py` |
| `_run_download()` | `presentation/runners.py` |
| `_run_preprocess()` | `presentation/runners.py` |
| `_run_train()` | `presentation/runners.py` |
| `_run_remote_train()` | `presentation/runners.py` |
| `_run_vertex_submit()` | `presentation/runners.py` |
| `_run_vertex_download()` | `presentation/runners.py` |
| `_run_update_source_dataset_pipeline()` | `presentation/runners.py` |
| `_run_push_notebook_pipeline()` | `presentation/runners.py` |
| `_run_inference()` | `presentation/runners.py` |
| `_ensure_cloud_config()` | `presentation/cloud_config.py` |
| `_load_trainer_cfgs_safe()` | `presentation/cloud_config.py` |
| `_resolve_manifest_path()` | `presentation/cloud_config.py` |
| Kaggle 認証ボイラープレート (3箇所) | `presentation/kaggle_auth.py` |
| `main()` 内の if/elif 分岐 | `presentation/registry.py` |

### Import 更新

- `tests/test_main_resolve.py`: `from src.main import _resolve_manifest_path` → `from src.presentation.cloud_config import _resolve_manifest_path`

## ステップ

1. lefthook / CI が既存テストを自動実行することを確認
2. テスト: `test_presentation_modules.py` — 新モジュールが正しくインポートでき、dispatch が動作すること
3. 実装: モジュール分割 + main.py スリム化
4. 既存テスト全パス確認
5. PR 作成
