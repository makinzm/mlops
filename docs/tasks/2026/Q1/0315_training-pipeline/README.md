# Training パイプライン 実装計画

## 背景

preprocess usecase で生成した fold 構造の parquet を入力として、
モデルを fold ごとに学習・評価し OOF 予測・モデル成果物を `models/` に保存する。

LightGBM を最初の実装対象とし、PyTorch を同一 Protocol で後から追加できる設計にする。

---

## 設計方針

### PyTorch アーキテクチャはコードで書く、YAML はクラスを参照するだけ

MobileNet / ResNet / GNN / 自作など多様なモデルを YAML で表現しきるのは不可能。
アーキテクチャの定義は Python コードに閉じ、YAML は以下のみ持つ:

- どのモデルクラスを使うか（`model.class` に import パスを書く）
- 共通の訓練パラメータ（optimizer / loss / scheduler / environment）

```
YAML の責務:          「何で学習するか」（optimizer, loss, epochs, device）
Python コードの責務:  「何を学習するか」（アーキテクチャ定義）
```

### サンプル重みの優先順位

```
sample_weight_col（行ごと） > class_weight（クラスごと手動） > is_unbalance（自動）
```

---

## conf 設計

### `conf/usecase/train.yaml`

```yaml
# @package _global_
usecase: train
# pipeline: 省略 → competition/train/ 配下の全 yaml を実行
# pipeline: lgbm → lgbm.yaml だけ実行
```

### `conf/competition/titanic/train/lgbm.yaml`

```yaml
# @package _global_
usecase: train
job_id: titanic_lgbm
trainer: lgbm

# ── 入力 ──────────────────────────────────────────
# "latest" を指定すると output_dir/{job_id}/ 配下の最新タイムスタンプを自動選択
preprocess_output_dir: data/2026/Q1/processed/titanic_preprocess/titanic_preprocess/latest/train_out
target_col: Survived
feature_cols:
  - Pclass
  - Age
  - SibSp
  - Parch
  - Fare
  - FamilySize
categorical_feature:
  - Sex
  - Embarked

# ── サンプル重み ─────────────────────────────────
# null: 重みなし / カラム名: そのカラムをサンプル重みに使用
sample_weight_col: null

# ── 損失 / 評価 ──────────────────────────────────
loss:
  objective: binary
  metric: auc
  is_unbalance: false           # true: lgbm が自動でクラス重みを調整
  class_weight: null            # {0: 1.0, 1: 3.0} のように手動指定

# ── LightGBM ハイパーパラメータ ──────────────────
lgbm:
  num_leaves: 31
  max_depth: -1
  learning_rate: 0.05
  n_estimators: 1000
  feature_fraction: 0.9
  bagging_fraction: 0.8
  bagging_freq: 5
  min_child_samples: 20
  reg_alpha: 0.0
  reg_lambda: 0.0
  verbose: -1
  early_stopping_rounds: 50

# ── モデルレポート ────────────────────────────────
report:
  n_error_samples: 50          # 当たり外れサンプリング: 各カテゴリ N 件

# ── 学習環境 ──────────────────────────────────────
environment:
  device: cpu
  n_jobs: -1

# ── ログ ─────────────────────────────────────────
logging:
  eval_freq: 100
  save_importance: true

output_dir: models/${competition.name}
seed: 42
```

### `conf/competition/titanic/train/nn.yaml`（将来テンプレート）

```yaml
# @package _global_
usecase: train
job_id: titanic_nn
trainer: pytorch

preprocess_output_dir: data/2026/Q1/processed/titanic_preprocess/titanic_preprocess/latest/train_out
target_col: Survived
feature_cols: [Pclass, Age, SibSp, Parch, Fare, FamilySize]
categorical_feature: [Sex, Embarked]
sample_weight_col: null

# ── モデルクラス参照 ─────────────────────────────
# アーキテクチャの定義は Python コードに書く。
# YAML はどのクラスを使うかだけを指定する。
#
# 例）
#   自作 MLP:     src.models.titanic_mlp.TitanicMLP
#   ResNet:       torchvision.models.resnet18
#   GNN:          src.models.titanic_gnn.TitanicGNN
model:
  class: src.models.titanic_mlp.TitanicMLP

# ── データセット / データローダー ─────────────────
dataset:
  type: tabular                 # tabular / image / text
  batch_size: 256
  num_workers: 4
  shuffle_train: true
  cat_encoding: embedding       # embedding / onehot

# ── 損失関数 ──────────────────────────────────────
loss:
  name: bce_with_logits
  pos_weight: null              # 正例への重み（スカラー）
  class_weight: null            # クラスごとの重みリスト
  sample_weight_col: null
  reduction: mean
  metric: auc

# ── Optimizer ────────────────────────────────────
optimizer:
  name: adamw                   # adam / adamw / sgd
  lr: 1.0e-3
  weight_decay: 1.0e-4

# ── スケジューラー ───────────────────────────────
scheduler:
  name: cosine_annealing        # cosine_annealing / reduce_on_plateau / none
  T_max: 100
  eta_min: 1.0e-6

# ── 学習ループ ────────────────────────────────────
trainer_params:
  epochs: 100
  early_stopping_rounds: 10
  gradient_clip: 1.0
  eval_freq: 5

# ── モデルレポート ────────────────────────────────
report:
  n_error_samples: 50

# ── 学習環境 ──────────────────────────────────────
environment:
  device: cuda                  # cpu / cuda / mps
  n_jobs: 4
  mixed_precision: true

output_dir: models/${competition.name}
seed: 42
```

---

## ディレクトリ・ファイル構成

```
src/
├── domain/
│   └── model/
│       ├── __init__.py
│       └── trainer.py              # Trainer Protocol / FoldResult / TrainResult
├── usecase/
│   └── training/
│       ├── __init__.py
│       ├── train.py                # TrainUseCase
│       └── trainer_loader.py       # load_trainer_cfgs
└── infrastructure/
    └── trainer/
        ├── __init__.py
        ├── factory.py
        └── lgbm.py                 # LightGBMTrainer

models/
└── titanic/
    └── titanic_lgbm/
        └── {timestamp}/
            ├── .gitignore          # 実行時生成
            ├── train_result.yaml   # git 管理
            ├── README.md           # git 管理（CV サマリ + ツリー）
            └── fold_0/
                ├── model.txt            # gitignore 対象
                ├── oof_train.parquet    # gitignore 対象
                ├── error_analysis.parquet  # gitignore 対象（後述）
                └── feature_importance.parquet  # gitignore 対象
```

---

## Domain Protocol 設計

```python
@dataclass
class FoldResult:
    fold_idx: int
    train_score: float
    valid_score: float
    metric: str
    model_path: Path
    oof_path: Path
    error_analysis_path: Path       # 予測の当たり外れサンプリング
    feature_importance_path: Path | None
    n_train: int
    n_valid: int
    best_iteration: int | None = None
    feature_importances: dict[str, float] = field(default_factory=dict)

@dataclass
class TrainResult:
    job_id: str
    timestamp: str
    commit_hash: str
    trainer_type: str
    output_dir: Path
    fold_results: list[FoldResult]
    cv_mean_score: float
    cv_std_score: float
    metric: str
    seed: int
    trainer_fallback: bool = False
    trainer_requested: str | None = None

class Trainer(Protocol):
    def fit_folds(
        self,
        preprocess_output_dir: Path,
        output_dir: Path,
        cfg: dict[str, Any],
    ) -> TrainResult: ...
```

---

## モデルレポート設計（予測の当たり外れサンプリング）

### 出力ファイル: `fold_{N}/error_analysis.parquet`

各 fold の validation セットに対して以下を記録:

| カラム | 内容 |
|--------|------|
| 全特徴量カラム | 元データの値 |
| `target` | 正解ラベル |
| `predicted_proba` | モデルの予測確率 |
| `predicted_label` | 閾値 0.5 で変換したラベル |
| `is_correct` | 正解かどうか |
| `error_magnitude` | `abs(predicted_proba - target)` |
| `sample_type` | 後述の 4 分類 |

### サンプリング戦略（`report.n_error_samples: 50`）

各 fold で以下 4 種類を N 件ずつサンプリングして記録:

```
True Positive  (正解=1, 予測=1): 高確率で正例を当てたサンプル
                                  → モデルが得意とするパターン

True Negative  (正解=0, 予測=0): 高確率で負例を当てたサンプル
                                  → モデルが得意とするパターン

False Positive (正解=0, 予測=1): 負例を正例と誤ったサンプル
                                  → モデルが混乱するパターン（特徴量を見返す）

False Negative (正解=1, 予測=0): 正例を負例と誤ったサンプル
                                  → モデルが見逃しやすいパターン
```

**ソート基準**:
- 当たり系（TP/TN）: `predicted_proba` の確信度が高い順（error_magnitude 小さい順）
- 外れ系（FP/FN）: `error_magnitude` が大きい順（最も自信を持って間違えたもの）

→ 後から `pd.read_parquet("fold_0/error_analysis.parquet")` で読み込んで確認可能。

### README.md のモデルレポート部分

```markdown
# Train Result — `titanic_lgbm`

- commit: `abc123`
- trainer: lgbm
- metric: auc
- **CV score: 0.8542 ± 0.0123**

## Fold Scores

| Fold | Train AUC | Valid AUC | Best Iter | n_train | n_valid |
|------|-----------|-----------|-----------|---------|---------|
| 0    | 0.9123    | 0.8621    | 234       | 712     | 179     |
| 1    | 0.9201    | 0.8534    | 289       | 712     | 179     |

## Prediction Error Sample (Fold 0)

### False Negative — モデルが見逃した生存者（上位 5 件）

| PassengerId | Pclass | Age | Fare | FamilySize | predicted_proba | error_magnitude |
|-------------|--------|-----|------|------------|-----------------|-----------------|
| 123         | 1      | 28  | 80.0 | 1          | 0.12            | 0.88            |
...

### False Positive — モデルが誤って生存と予測した非生存者（上位 5 件）

| PassengerId | Pclass | Age | Fare | FamilySize | predicted_proba | error_magnitude |
|-------------|--------|-----|------|------------|-----------------|-----------------|
| 456         | 2      | 35  | 30.0 | 3          | 0.91            | 0.91            |
...
```

---

## LightGBMTrainer 設計

```
fit_folds():
  for each fold_N/:
    1. train.parquet / test.parquet を読み込み
    2. サンプル重みを構築（sample_weight_col > class_weight > is_unbalance）
    3. lgb.Dataset を作成（weight 付き）
    4. lgb.train() — early stopping、eval_freq ごとにログ出力
    5. fold_{N}/model.txt に保存
    6. validation セットで予測 → oof_train.parquet に保存
    7. error_analysis_parquet を生成（4 種サンプリング）
    8. feature importance を parquet に保存

  CV スコアを集計 → TrainResult を返す
```

---

## PyTorch Trainer 設計（将来実装）

### モデルクラスのインターフェース規約

YAML の `model.class` から動的にインポートするため、
すべてのモデルクラスは以下のクラスメソッドを持つ:

```python
class TitanicMLP(nn.Module):
    """YAML の model.class に指定して使うモデル。

    from_data_meta() で入力次元などのデータ依存パラメータを注入する。
    """

    def __init__(self, input_dim: int, ...):
        ...

    @classmethod
    def from_data_meta(cls, data_meta: dict[str, Any]) -> "TitanicMLP":
        """データのメタ情報（特徴量数など）から初期化する。

        PyTorchTrainer からはこのメソッドだけを呼ぶ。
        アーキテクチャの詳細は Python コード内に閉じる。
        """
        return cls(input_dim=data_meta["n_features"], ...)
```

`data_meta` には特徴量数・カテゴリ変数の cardinality などデータに依存する情報を含める。
モデルの内部構造（hidden_dims, dropout など）は Python コード内に書く。

### ローダー

```python
def _load_model_class(class_path: str) -> type[nn.Module]:
    """'src.models.titanic_mlp.TitanicMLP' 形式で動的にインポートする。"""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
```

---

## `preprocess_output_dir: latest` の自動解決

```python
def resolve_preprocess_dir(path_str: str) -> Path:
    """'latest' を含むパスを最新タイムスタンプに解決する。"""
    # .../titanic_preprocess/latest/train_out
    # → .../titanic_preprocess/{最新 timestamp}/train_out
    parts = Path(path_str).parts
    latest_idx = [i for i, p in enumerate(parts) if p == "latest"]
    if not latest_idx:
        return Path(path_str)

    idx = latest_idx[0]
    parent = Path(*parts[:idx])
    suffix = Path(*parts[idx + 1:]) if len(parts) > idx + 1 else Path()

    candidates = sorted(parent.iterdir(), key=lambda p: p.name, reverse=True)
    dirs = [c for c in candidates if c.is_dir() and c.name != "latest"]
    if not dirs:
        raise ValueError(f"No processed directory found under {parent}")

    return dirs[0] / suffix
```

---

## 出力 .gitignore の内容（models/ 向け）

```
*
!.gitignore
!*.yaml
!*.md
!*/
```

モデルバイナリ（`*.txt`, `*.pkl`, `*.pt`）・parquet は除外。
`train_result.yaml` と `README.md` だけ git に残す。

---

## 実装順序（TDD サイクル）

| Phase | 対象 | 内容 |
|-------|------|------|
| 1 | Domain | `FoldResult` / `TrainResult` / `Trainer` Protocol |
| 2 | trainer_loader | `load_trainer_cfgs()` |
| 3 | TrainUseCase | Mock Trainer で単体テスト（.gitignore・manifest・README） |
| 4 | LightGBMTrainer | fold parquet で実際に学習 + error_analysis 生成 |
| 5 | TrainerFactory | |
| 6 | conf + main.py | `conf/usecase/train.yaml` を先に作成してから main.py に分岐 |
| 7 | docs | `docs/manual/0500_train.md` |

## 依存ライブラリ

```bash
uv add lightgbm
```
