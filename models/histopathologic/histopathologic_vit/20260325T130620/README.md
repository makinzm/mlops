# Train Result — `histopathologic_vit`

- commit: `8c8da25fc49fa75b4128f0edeca148ed6085fe34`
- trainer: vision
- metric: accuracy
- **CV score: 0.8327 ± 0.0014**

## Fold Scores

| Fold | Train ACCURACY | Valid ACCURACY | Best Iter | n_train | n_valid |
|------|----|----|----|----|----|
| 0 | 0.8222 | 0.8311 | - | 14667 | 7334 |
| 1 | 0.8247 | 0.8326 | - | 14667 | 7334 |
| 2 | 0.8285 | 0.8344 | - | 14668 | 7333 |

## Output Files

```
20260325T130620/
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
