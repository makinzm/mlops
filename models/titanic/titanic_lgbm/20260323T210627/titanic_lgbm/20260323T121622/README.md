# Train Result — `titanic_lgbm`

- commit: `unknown`
- trainer: lgbm
- metric: auc
- **CV score: 0.7863 ± 0.0348**

## Fold Scores

| Fold | Train AUC | Valid AUC | Best Iter | n_train | n_valid |
|------|----|----|----|----|----|
| 0 | 0.8376 | 0.8223 | 32 | 712 | 179 |
| 1 | 0.8949 | 0.8301 | 105 | 713 | 178 |
| 2 | 0.8839 | 0.7536 | 70 | 713 | 178 |
| 3 | 0.8472 | 0.7446 | 29 | 713 | 178 |
| 4 | 0.9135 | 0.7811 | 123 | 713 | 178 |

## Output Files

```
20260323T121622/
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
