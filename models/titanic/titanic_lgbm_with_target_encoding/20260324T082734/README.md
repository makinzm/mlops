# Train Result — `titanic_lgbm_with_target_encoding`

- commit: `b9cd0575796d6399df6eefa5d08dc4e820cfa58b`
- trainer: lgbm
- metric: auc
- **CV score: 0.8884 ± 0.0214**

## Fold Scores

| Fold | Train AUC | Valid AUC | Best Iter | n_train | n_valid |
|------|----|----|----|----|----|
| 0 | 0.9449 | 0.9111 | 80 | 712 | 179 |
| 1 | 0.9652 | 0.9043 | 120 | 713 | 178 |
| 2 | 0.9552 | 0.8543 | 78 | 713 | 178 |
| 3 | 0.9812 | 0.8729 | 176 | 713 | 178 |
| 4 | 0.9404 | 0.8996 | 55 | 713 | 178 |

## Output Files

```
20260324T082734/
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
