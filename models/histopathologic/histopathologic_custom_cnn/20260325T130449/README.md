# Train Result — `histopathologic_custom_cnn`

- commit: `8c8da25fc49fa75b4128f0edeca148ed6085fe34`
- trainer: vision
- metric: accuracy
- **CV score: 0.8506 ± 0.0088**

## Fold Scores

| Fold | Train ACCURACY | Valid ACCURACY | Best Iter | n_train | n_valid |
|------|----|----|----|----|----|
| 0 | 0.8302 | 0.8601 | - | 14667 | 7334 |
| 1 | 0.8304 | 0.8526 | - | 14667 | 7334 |
| 2 | 0.8286 | 0.8389 | - | 14668 | 7333 |

## Output Files

```
20260325T130449/
├── fold_0
│   ├── error_analysis.parquet
│   ├── model.pt
│   └── oof_train.parquet
├── fold_1
│   ├── error_analysis.parquet
│   ├── model.pt
│   └── oof_train.parquet
├── fold_2
│   ├── error_analysis.parquet
│   ├── model.pt
│   └── oof_train.parquet
└── train_result.yaml
```
