# Vision Model Training 手動操作手順

## 前提条件

- `uv sync` で依存パッケージがインストール済み
- 前処理済みデータが `data/2026/Q1/processed/` に存在する
  - 各 fold ディレクトリ (`fold_0/`, `fold_1/`, ...) に `train.parquet` と `test.parquet` がある
  - DataFrame に `image_path` カラム（画像ファイルの絶対パス）と `label` カラム（分類ラベル）が含まれる

## 1. 学習設定ファイルの作成

`conf/competition/<competition_name>/training/` に YAML ファイルを作成する。

例: `conf/competition/titanic/training/vision_resnet50.yaml`

```yaml
# @package _global_
usecase: train
job_id: titanic_vision_resnet50
trainer:
  type: vision

# ── 入力 ──────────────────────────────────────────
preprocess_output_dir: data/2026/Q1/processed/titanic_preprocess/latest/train_out
target_col: label
image_path_col: image_path

# ── backbone ──────────────────────────────────────
# 利用可能: resnet18, resnet34, resnet50, vit_b_16, vit_b_32,
#           mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large, simple_cnn
backbone:
  name: resnet50
  pretrained: true
  image_size: 224

# ── 学習パラメータ ────────────────────────────────
training:
  num_epochs: 20
  batch_size: 32
  learning_rate: 0.001
  num_workers: 4

num_classes: 2

# ── レポート ──────────────────────────────────────
report:
  n_error_samples: 50

output_dir: ${oc.env:MLOPS_MODEL_DIR,models/${competition.name}}
seed: 42
```

## 2. 学習の実行

```bash
# 単一レシピ指定
uv run python -m src usecase=train recipe=vision_resnet50

# 全レシピ実行（training/ 配下の全 YAML）
uv run python -m src usecase=train
```

### 出力ディレクトリ

```
models/<competition>/<job_id>/
├── .gitignore
└── <YYYYMMDDTHHMMSS>/
    ├── train_result.yaml
    ├── README.md
    └── fold_0/
        ├── model.pt
        ├── oof_train.parquet
        └── error_analysis.parquet
```

## 3. 推論の実行

推論設定ファイルで `inferencer_type: vision` を指定する。

例: `conf/competition/titanic/inference/vision_ensemble.yaml`

```yaml
# @package _global_
usecase: inference
job_id: titanic_vision_inference
inferencer_type: vision

test_path: data/2026/Q1/processed/titanic_preprocess/latest/test_out/test.parquet
feature_cols:
  - image_path

passenger_id_col: PassengerId

models:
  - models/titanic/titanic_vision_resnet50/latest

ensemble: mean
output_dir: data/2026/Q1/inference/${competition.name}/${job_id}
submission:
  threshold: 0.5
seed: 42
```

```bash
uv run python -m src usecase=inference recipe=vision_ensemble
```

## 4. GradCAM 分析

学習済みモデルに対して GradCAM ヒートマップを生成する。

### 設定変更

`conf/usecase/gradcam.yaml` のパラメータを CLI で上書きする:

```bash
uv run python -m src usecase=gradcam \
  model_path=models/titanic/titanic_vision_resnet50/latest/fold_0/model.pt \
  image_dir=data/2026/Q1/raw/titanic/images \
  output_dir=data/2026/Q1/gradcam/titanic
```

### 出力

```
data/2026/Q1/gradcam/titanic/
├── .gitignore
├── metainfo.yaml
├── README.md
├── gradcam_image_0001.png
├── gradcam_image_0002.png
└── ...
```

## 5. 利用可能な backbone 一覧

| backbone 名 | パラメータ数 | 推奨 image_size | 特徴 |
|-------------|-------------|----------------|------|
| `simple_cnn` | ~5K | 32 | テスト・軽量実験用 |
| `resnet18` | 11.7M | 224 | 軽量 ResNet |
| `resnet34` | 21.8M | 224 | 中量 ResNet |
| `resnet50` | 25.6M | 224 | 標準 ResNet |
| `vit_b_16` | 86.6M | 224 | Vision Transformer (patch 16) |
| `vit_b_32` | 88.2M | 224 | Vision Transformer (patch 32) |
| `mobilenet_v2` | 3.4M | 224 | モバイル向け軽量 |
| `mobilenet_v3_small` | 2.5M | 224 | MobileNetV3 Small |
| `mobilenet_v3_large` | 5.5M | 224 | MobileNetV3 Large |

## 6. seed 再現性

以下の設定により再現性が保証される:
- `torch.manual_seed(seed)`: PyTorch の乱数シード
- `torch.backends.cudnn.deterministic = True`: cuDNN の決定性
- `torch.backends.cudnn.benchmark = False`: cuDNN ベンチマーク無効化
- `numpy.random.seed(seed)`: NumPy の乱数シード
- DataLoader の `generator=torch.Generator().manual_seed(seed)`: シャッフルの再現性
