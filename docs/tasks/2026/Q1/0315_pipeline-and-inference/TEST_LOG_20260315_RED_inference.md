# TEST_LOG: RED phase — Inference テスト

**日時**: 2026-03-15
**フェーズ**: RED (テスト実装→失敗確認)
**対象テストファイル**:
- `tests/usecase/inference/test_ensemble_strategies.py`
- `tests/usecase/inference/test_inference_usecase.py`
- `tests/infrastructure/inference/test_lgbm_inferencer.py`

## 実行コマンド

```bash
uv run pytest tests/usecase/inference/ tests/infrastructure/inference/ -v
```

## 結果サマリー

```
3 errors during collection
```

実装がまだ存在しないため、全テストファイルが import エラーで失敗している。期待通りの RED 状態。

## エラー詳細

```
tests/usecase/inference/test_ensemble_strategies.py
  ModuleNotFoundError: No module named 'src.usecase.inference'

tests/usecase/inference/test_inference_usecase.py
  ModuleNotFoundError: No module named 'src.usecase.inference'

tests/infrastructure/inference/test_lgbm_inferencer.py
  ModuleNotFoundError: No module named 'src.infrastructure.inference'
```

## 新規テストケース一覧

### TestMeanStrategy (2件)
1. `test_mean_strategy` — 2予測の単純平均が正しいこと
2. `test_mean_strategy_single_pred` — 1予測のみの場合はそのまま返すこと

### TestWeightedMeanStrategy (3件)
3. `test_weighted_mean_strategy` — weights=[0.6, 0.4] で加重平均になること
4. `test_weighted_mean_normalizes_weights` — weights 合計が 1.0 でなくても正規化されること
5. `test_weighted_mean_raises_on_weight_mismatch` — 数が合わない場合 ValueError

### TestRankAverageStrategy (2件)
6. `test_rank_average_strategy` — rank 変換後に平均されること
7. `test_rank_average_is_robust_to_outlier` — 外れ値に robust であること

### TestInferenceUseCaseRun (5件)
8. `test_run_creates_submission_csv` — submission.csv が生成されること
9. `test_run_submission_has_correct_columns` — PassengerId, Survived 列を持つこと
10. `test_run_uses_ensemble_strategy` — MockInferencer の predict_folds が呼ばれること
11. `test_run_records_commit_hash` — metainfo.yaml に commit_hash が含まれること
12. `test_run_generates_gitignore` — per-directory .gitignore が生成されること

### TestLightGBMInferencer (4件)
13. `test_predict_folds_returns_ndarray` — shape=(n_test,) の ndarray が返ること
14. `test_predict_folds_values_in_probability_range` — 予測値が [0, 1] 範囲内
15. `test_predict_folds_averages_over_folds` — 複数 fold の平均が正しいこと
16. `test_predict_folds_raises_when_no_model_dir` — fold なしで ValueError

## 次のステップ

GREEN フェーズ: 以下を実装する。

1. `src/usecase/inference/ensemble_strategies.py` — MeanStrategy, WeightedMeanStrategy, RankAverageStrategy
2. `src/usecase/inference/inference.py` — InferenceUseCase
3. `src/infrastructure/inference/lgbm_inferencer.py` — LightGBMInferencer
4. `conf/usecase/inference.yaml` — Hydra usecase config
5. `conf/competition/titanic/inference/titanic_ensemble.yaml` — 競技固有設定
