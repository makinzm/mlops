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
#           mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large,
#           simple_cnn, custom_cnn
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

### 入力バリデーション

学習開始前に自動的に以下が検証されます:

- fold ディレクトリの存在
- parquet ファイルの存在と必要カラム
- 画像パスの存在（サンプルチェック）
- ラベルの範囲（0 から num_classes-1）
- 画像サイズとモデル期待サイズの比較
- backbone 出力次元の整合性

エラー時は修正方法を含むメッセージが表示されます:

```
[image_path] fold_0/train.parquet: 3/10 枚の画像が見つかりません。
例: /data/images/missing.png
修正: 画像ファイルのパスが正しいか確認してください。
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
| `custom_cnn` | 設定依存 | 設定依存 | Config-driven カスタム CNN |

## 6. Custom CNN の作り方

### 方法 A: Config で定義

```yaml
backbone:
  name: custom_cnn
  image_size: 64
  custom_cnn:
    layers:
      - {out_channels: 32, kernel_size: 3, padding: 1, batch_norm: true, pool: max}
      - {out_channels: 64, kernel_size: 3, padding: 1, batch_norm: true, pool: null}
      - {out_channels: 64, kernel_size: 3, padding: 1, batch_norm: true, pool: max}
    skip_connections:
      - {type: residual, from_layer: 0, to_layer: 1}
    adaptive_pool_size: 1
```

#### ConvBlockConfig パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `out_channels` | (必須) | 出力チャネル数 |
| `kernel_size` | 3 | カーネルサイズ |
| `stride` | 1 | ストライド |
| `padding` | 1 | パディング |
| `activation` | "relu" | 活性化関数（"relu", "silu", "gelu"）|
| `batch_norm` | true | BatchNorm を使うか |
| `pool` | "max" | プーリング（"max", "avg", null） |

#### SkipConnectionConfig パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `type` | (必須) | "residual" or "inverted_bottleneck" |
| `from_layer` | 0 | 接続元レイヤーインデックス |
| `to_layer` | 1 | 接続先レイヤーインデックス |
| `expansion_factor` | 1 | inverted_bottleneck の拡張係数 |

#### ResNet ライクな構成例

```yaml
backbone:
  name: custom_cnn
  image_size: 64
  custom_cnn:
    layers:
      - {out_channels: 64, kernel_size: 7, stride: 2, padding: 3, pool: max}
      - {out_channels: 64, kernel_size: 3, pool: null}
      - {out_channels: 64, kernel_size: 3, pool: null}
      - {out_channels: 128, kernel_size: 3, pool: max}
    skip_connections:
      - {type: residual, from_layer: 0, to_layer: 1}
      - {type: residual, from_layer: 1, to_layer: 2}
```

#### MobileNet ライクな構成例（Inverted Bottleneck）

```yaml
backbone:
  name: custom_cnn
  image_size: 64
  custom_cnn:
    layers:
      - {out_channels: 16, kernel_size: 3}
      - {out_channels: 24, kernel_size: 3, pool: null}
      - {out_channels: 32, kernel_size: 3}
    skip_connections:
      - {type: inverted_bottleneck, from_layer: 0, to_layer: 1, expansion_factor: 6}
```

### 方法 B: Python コードで登録

```python
from torch import nn
from src.infrastructure.trainer.backbone_registry import register_backbone

def my_custom_backbone(pretrained: bool) -> tuple[nn.Module, int]:
    """カスタム backbone を返す。(module, output_features) のタプル。"""
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
    )
    return model, 64

register_backbone("my_backbone", my_custom_backbone)
```

その後、設定ファイルで `backbone.name: my_backbone` を指定して使用できます。

## 7. Data Augmentation

Albumentations を使った Data Augmentation に対応しています。

### 設定例

```yaml
augmentation:
  train:
    - {name: HorizontalFlip, probability: 0.5}
    - {name: RandomBrightnessContrast, probability: 0.3,
       params: {brightness_limit: 0.2}}
    - {name: ShiftScaleRotate, probability: 0.3}
    - {name: CoarseDropout, probability: 0.2,
       params: {max_holes: 8, max_height: 8, max_width: 8}}
  valid: []  # 空 = Resize + Normalize のみ
```

### 利用可能な transform

Albumentations の全 transform が利用可能です。
`name` には Albumentations のクラス名を、`params` にはコンストラクタ引数を指定します。

よく使う transform:

| name | 説明 | 主なパラメータ |
|------|------|--------------|
| `HorizontalFlip` | 左右反転 | - |
| `VerticalFlip` | 上下反転 | - |
| `RandomBrightnessContrast` | 明度・コントラスト | brightness_limit, contrast_limit |
| `ShiftScaleRotate` | 移動・拡縮・回転 | shift_limit, scale_limit, rotate_limit |
| `CoarseDropout` | ランダムマスク | max_holes, max_height, max_width |
| `GaussNoise` | ガウスノイズ | var_limit |
| `Blur` | ぼかし | blur_limit |

### Albumentations 未インストール時

torchvision の Resize + ToTensor + Normalize にフォールバックします（警告表示）。

## 8. 画像前処理（Preprocessing）

前処理パイプラインで `image` resolver を使って画像のバリデーションとメタデータ取得ができます。

### 設定例

```yaml
# conf/competition/<name>/preprocess/image_base.yaml
steps:
  - resolver: image
    method: validate_images
    kwargs:
      column: image_path

  - resolver: image
    method: create_image_metadata
    kwargs:
      column: image_path
```

### メソッド一覧

| メソッド | 説明 | 追加カラム |
|---------|------|----------|
| `validate_images` | 画像パスの存在確認 | `__image_valid__` (bool) |
| `create_image_metadata` | 画像のサイズ・チャネル取得 | `__image_width__`, `__image_height__`, `__image_channels__` |

## 9. seed 再現性

以下の設定により再現性が保証されます（`torch_utils/seed.py` で一括管理）:

- `torch.manual_seed(seed)`: PyTorch の乱数シード
- `torch.backends.cudnn.deterministic = True`: cuDNN の決定性
- `torch.backends.cudnn.benchmark = False`: cuDNN ベンチマーク無効化
- `numpy.random.seed(seed)`: NumPy の乱数シード
- DataLoader の `generator=torch.Generator().manual_seed(seed)`: シャッフルの再現性
