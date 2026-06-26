# 音声コンペ汎用基盤の導入（Phase 1）

## 背景

姉妹リポジトリ `bird-clef-2026`（BirdCLEF 2026 で実際に使用）には音声処理の実装が揃っている。
そのうち BirdCLEF 固有のドメインロジック（鳥の種名・Perch embedding 等）を除いた、
音声コンペ全般で使える汎用基盤を mlops に移植する。

ユーザーとの相談によりスコープは **Phase 1（汎用基盤のみ）** に確定。
embedding / Perch / TensorFlow 系（`PerchEmbedder`, `LogitsMapper`,
`ExtractEmbeddingsUseCase`, `EmbeddingHeadTrainer`, `EmbeddingHeadInferencer`）は対象外。

## スコープ

### 実装対象

| ファイル | 内容 |
|---|---|
| `src/domain/data/audio.py` | `AudioSample` dataclass, `AudioLoader` Protocol, `SpectrogramConfig` dataclass, `SpectrogramTransformer` Protocol |
| `src/infrastructure/audio/soundfile_loader.py` | `SoundfileLoader`（`AudioLoader` 実装） |
| `src/infrastructure/audio/mel_spectrogram.py` | `MelSpectrogramTransformer`（`SpectrogramTransformer` 実装） |
| `src/infrastructure/trainer/torch_utils/audio_augmentation.py` | `spec_augment()`, `mixup()`（public 化） |
| `src/infrastructure/trainer/audio_trainer.py` | `AudioTrainer`（`Trainer` Protocol 実装。backbone/num_classes は cfg 経由） |
| `pyproject.toml` | `soundfile>=0.12` を dependencies に追加 |
| `mille.toml` | `src_infrastructure` / `tests_infrastructure` の `external_allow` に `soundfile` を追加 |
| `conf/competition/audio_example/` | 動作確認用の最小サンプル設定（ダミー、実データ不要） |
| `docs/manual/audio-foundation.md` | 動かし方マニュアル |

### 対象外（Phase 2 以降）

- Perch embedding / TensorFlow 系
- BirdCLEF 固有のラベルエンコーディング（`LabelEncoder`, 234-class taxonomy）
- 推論（`AudioInferencer`）— Phase 1 は学習基盤のみ

## 設計判断

### ドメイン層の命名

`bird-clef-2026/src/domain/data/audio.py` は元々ドメイン固有の命名が混入していないため、
ほぼそのまま移植する。`SpectrogramConfig` は `bird-clef-2026/src/domain/data/label.py` にあるが、
ラベル関連クラス（`LabelEncoder`）と同居しているため、mlops では
`src/domain/data/audio.py` に集約する（ファイル名とクラスの対応を一致させる）。

### AudioTrainer の汎用化

元実装は EfficientNet-B0 固定・234 クラス固定だったが、以下のように cfg 経由にする:

- `model.backbone`: backbone 名（`efficientnet_b0` のみ対応、将来拡張可能なように `_build_audio_model` 関数で分岐）
- `model.num_classes`: クラス数（デフォルト値なし、必須指定にする）
- `model.pretrained`: pretrained 重みを使うか

`AudioTrainer.__init__` は `VisionTrainer` と同じ DI パターン
（`SeedFixer` をコンストラクタで受け取り、デフォルトは `TorchSeedFixer`）に揃える。

### spec_augment / mixup の public 化

`bird-clef-2026` の `_spec_augment` / `_mixup` は private 関数として `audio_trainer.py` 内に
あったが、mlops では既存の `torch_utils/augmentation.py`（vision 用、albumentations ベース）
と役割が異なるため、新規ファイル `torch_utils/audio_augmentation.py` に分離し、
`spec_augment()` / `mixup()` として public 化する。

### サンプル conf

`conf/competition/audio_example/` を新設し、`titanic` / `histopathologic` と同じ構造
（`{competition}.yaml` + `training/*.yaml`）に揃える。実データは用意せず、設定の形のみ示す。

## 品質ゲート（DoD）

実装完了後、DA レビュー前に以下を実行し、全て clean であることを確認する:

1. `uv run pytest`
2. `uv run ruff check .` / `uv run ruff format --check .`
3. `uv run ty check src/ tests/ --python .venv`
4. `uv run mille check`
