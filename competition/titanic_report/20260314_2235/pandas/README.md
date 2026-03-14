# EDA Report — titanic [pandas]

Generated: 2026-03-14T22:35:36

## Files analyzed

### gender_submission.csv

Shape: 418 rows × 2 cols  |  Output files: 5

#### Schema

| Column | Type | Missing | Missing % |
|--------|------|--------:|----------:|
| PassengerId | int64 | 0 | 0.0% |
| Survived | int64 | 0 | 0.0% |

### test.csv

Shape: 418 rows × 11 cols  |  Output files: 14

#### Schema

| Column | Type | Missing | Missing % |
|--------|------|--------:|----------:|
| PassengerId | int64 | 0 | 0.0% |
| Pclass | int64 | 0 | 0.0% |
| Name | object | 0 | 0.0% |
| Sex | object | 0 | 0.0% |
| Age | float64 | 86 | 20.6% |
| SibSp | int64 | 0 | 0.0% |
| Parch | int64 | 0 | 0.0% |
| Ticket | object | 0 | 0.0% |
| Fare | float64 | 1 | 0.2% |
| Cabin | object | 327 | 78.2% |
| Embarked | object | 0 | 0.0% |

### train.csv

Shape: 891 rows × 12 cols  |  Output files: 15

#### Schema

| Column | Type | Missing | Missing % |
|--------|------|--------:|----------:|
| PassengerId | int64 | 0 | 0.0% |
| Survived | int64 | 0 | 0.0% |
| Pclass | int64 | 0 | 0.0% |
| Name | object | 0 | 0.0% |
| Sex | object | 0 | 0.0% |
| Age | float64 | 177 | 19.9% |
| SibSp | int64 | 0 | 0.0% |
| Parch | int64 | 0 | 0.0% |
| Ticket | object | 0 | 0.0% |
| Fare | float64 | 0 | 0.0% |
| Cabin | object | 687 | 77.1% |
| Embarked | object | 2 | 0.2% |

## Analyses run

- `basic_stats`
- `distributions`
- `missing_values`