# 音声コンペ汎用基盤の使い方

## 概要

`src/domain/data/audio.py` と `src/infrastructure/audio/`, `src/infrastructure/trainer/audio_trainer.py` が提供する汎用音声処理基盤の使い方を説明する。

BirdCLEF 2026 の実装から BirdCLEF 固有の部分（Perch embedding / 234-class taxonomy 等）を除いた、音声コンペ全般に使える基盤。

---

## 前提

`soundfile` は `uv sync` で自動インストールされる。追加の手順は不要。

---

## 1. 新しい音声コンペを追加する

### 1-1. competition 設定ファイルを作成する

`conf/competition/<competition_name>.yaml` を作成する。

```yaml
# conf/competition/birdclef-2027.yaml
# @package _global_
competition:
  name: birdclef-2027
```

### 1-2. training recipe を作成する

`conf/competition/<competition_name>/training/<recipe_name>.yaml` を作成する。
`audio_example` のファイルをコピーして編集するのが最も簡単。

```bash
cp conf/competition/audio_example/training/efficientnet_b0.yaml \
   conf/competition/birdclef-2027/training/efficientnet_b0.yaml
```

編集する主なパラメータ:

| キー | 説明 | 例 |
|---|---|---|
| `model.num_classes` | 分類クラス数 | `182` |
| `spectrogram.sample_rate` | サンプリングレート（Hz） | `32000` |
| `spectrogram.segment_seconds` | 1セグメントの長さ（秒） | `5.0` |
| `training.epochs` | エポック数 | `30` |
| `training.batch_size` | バッチサイズ | `32` |
| `augmentation.spec_augment` | SpecAugment を使うか | `true` |
| `augmentation.mixup` | Mixup を使うか | `true` |

---

## 2. 前処理出力の形式

`AudioTrainer.fit_folds` は以下の2ファイルを `preprocess_output_dir` から読む。

### manifest.json

```json
[
  {"file_path": "path/to/sample_0.pt", "label": [1.0, 0.0, 0.0]},
  {"file_path": "path/to/sample_1.wav", "label": [0.0, 1.0, 0.0]}
]
```

- `file_path` が `.pt` の場合: `torch.load` で読み込む。
  - 1D テンソル（波形）→ `MelSpectrogramTransformer` で変換する。
  - 2D テンソル（事前計算済みスペクトログラム）→ そのまま使う。
- `file_path` が `.wav` 等の場合: on-demand でスペクトログラムに変換する。

### cv_splits.json

```json
[
  {"train": [0, 1, 2, 3], "val": [4, 5]},
  {"train": [4, 5], "val": [0, 1, 2, 3]}
]
```

manifest のインデックスで fold を指定する。

---

## 3. 学習を実行する

```bash
uv run python -m src \
  usecase=train \
  competition=birdclef-2027 \
  recipe=efficientnet_b0
```

出力先は `cfg.output_dir`（デフォルト: `models/original/birdclef-2027/<timestamp>/`）。

各 fold のモデルは `fold_0/model.pt`, `fold_1/model.pt`, ... に保存される。

チェックポイントには以下が含まれる:

```python
{
    "model_state_dict": ...,
    "backbone": "efficientnet_b0",
    "num_classes": 182,
    "spectrogram_config": {...}
}
```

---

## 4. SpecAugment / Mixup の設定を変更する

`augmentation` セクションで on/off と強度を調整できる。

```yaml
augmentation:
  spec_augment: true
  freq_mask_param: 20   # 周波数マスクの最大幅（メルバンド数）
  time_mask_param: 40   # 時間マスクの最大幅（フレーム数）
  mixup: true
  mixup_alpha: 0.4      # Beta 分布パラメータ（大きいほど混合比が 0.5 に近づく）
```

---

## 5. サンプル設定で動作を確認する

実データなしで設定が正しくロードされるか確認するには:

```bash
uv run python -m src \
  usecase=train \
  competition=audio_example \
  recipe=efficientnet_b0 \
  --cfg job
```

`--cfg job` は Hydra の設定ダンプオプション。実際に学習は実行しない。
