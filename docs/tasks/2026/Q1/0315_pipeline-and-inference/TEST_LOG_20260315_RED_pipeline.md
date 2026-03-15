# TEST_LOG: RED phase — Pipeline テスト

**日時**: 2026-03-15
**フェーズ**: RED (テスト実装→失敗確認)
**対象テストファイル**: `tests/usecase/pipeline/test_pipeline_usecase.py`

## 実行コマンド

```bash
uv run pytest tests/usecase/pipeline/ -v
```

## 結果サマリー

```
1 error during collection
```

`src.usecase.pipeline` モジュールが未実装のため import エラー。期待通りの RED 状態。

## エラー詳細

```
tests/usecase/pipeline/test_pipeline_usecase.py:20: in <module>
    from src.usecase.pipeline.pipeline import PipelineUseCase
ModuleNotFoundError: No module named 'src.usecase.pipeline'
```

## 新規テストケース一覧 (5件)

| テスト名 | 検証内容 |
|---------|---------|
| `test_run_executes_steps_in_order` | steps が定義順に実行されること |
| `test_run_calls_each_runner_with_merged_cfg` | step.recipe が DI 先の cfg にマージされること |
| `test_run_stops_on_failure` | 1ステップ失敗で後続が止まること（fail-fast） |
| `test_run_raises_for_unknown_usecase` | 未知の usecase で ValueError |
| `test_run_single_step` | ステップが 1 つだけでも動作すること |

## 次のステップ

GREEN フェーズ: 以下を実装する。

1. `src/usecase/pipeline/pipeline.py` — PipelineUseCase
2. `conf/usecase/pipeline.yaml` — Hydra usecase config
3. `conf/competition/titanic/pipeline/all_after_download.yaml` — 競技固有パイプライン
4. `src/main.py` に `elif usecase_name == "pipeline":` ブロック追加
5. `src/main.py` の各 usecase 処理を `_run_preprocess()` / `_run_train()` / `_run_inference()` に切り出す
