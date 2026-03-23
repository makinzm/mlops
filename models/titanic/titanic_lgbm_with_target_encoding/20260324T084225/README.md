# Train Result — `titanic_lgbm_with_target_encoding`

- commit: `8e787af8e5ac79139029d9a746b266e36a8c4c28`
- trainer: lgbm
- metric: auc
- **CV score: 0.8780 ± 0.0231**

## Fold Scores

| Fold | Train AUC | Valid AUC | Best Iter | n_train | n_valid |
|------|----|----|----|----|----|
| 0 | 0.9420 | 0.9100 | 59 | 712 | 179 |
| 1 | 0.9408 | 0.8842 | 49 | 713 | 178 |
| 2 | 0.9118 | 0.8452 | 4 | 713 | 178 |
| 3 | 0.9391 | 0.8592 | 45 | 713 | 178 |
| 4 | 0.9450 | 0.8912 | 48 | 713 | 178 |

## Output Files

```
20260324T084225/
├── fold_0
│   ├── error_analysis.parquet
│   ├── feature_importance.parquet
│   ├── model.lgbm
│   └── oof_train.parquet
├── fold_1
│   ├── error_analysis.parquet
│   ├── feature_importance.parquet
│   ├── model.lgbm
│   └── oof_train.parquet
├── fold_2
│   ├── error_analysis.parquet
│   ├── feature_importance.parquet
│   ├── model.lgbm
│   └── oof_train.parquet
├── fold_3
│   ├── error_analysis.parquet
│   ├── feature_importance.parquet
│   ├── model.lgbm
│   └── oof_train.parquet
├── fold_4
│   ├── error_analysis.parquet
│   ├── feature_importance.parquet
│   ├── model.lgbm
│   └── oof_train.parquet
└── train_result.yaml
```
