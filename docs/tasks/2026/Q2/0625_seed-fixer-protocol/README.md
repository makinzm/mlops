# SeedFixer を Protocol として domain 層に抽象化

## 背景

`fix_seed()` は `src/infrastructure/trainer/torch_utils/seed.py` に直接実装されており、
`vision_trainer.py` が PyTorch 固有の実装に直接 import 依存している
（usecase/infrastructure 境界に Protocol が無い）。

姉妹リポジトリ `../bird-clef-2026` では `src/domain/seed.py`（Protocol）+
`src/infrastructure/shared/seed.py`（AllSeedFixer 実装）という形で抽象化しており、
将来 TensorFlow/JAX 等の別フレームワークを使う Trainer を追加する際に
domain/usecase 層を変更せず Infrastructure 実装を差し替えられる。

## やること

1. `src/domain/model/seed.py` に `SeedFixer` Protocol（`fix(seed: int) -> None`）を定義する
   （既存の `src/domain/model/trainer.py` の Protocol 定義スタイルに合わせる）。
2. `src/infrastructure/trainer/torch_utils/seed.py` に `TorchSeedFixer` クラスを追加し、
   既存の `fix_seed()` 関数をそのまま内部で呼ぶ（後方互換: 関数自体は削除しない。
   既存テスト `tests/infrastructure/trainer/torch_utils/test_seed.py` を変更しない）。
3. `vision_trainer.py` で `TorchSeedFixer` を Protocol 越しに使うように変更する。
4. テスト: `SeedFixer` Protocol を満たすことの確認、`TorchSeedFixer.fix()` が
   `fix_seed()` と同じ再現性を持つことの確認。

## 対象外（スコープ外）

- LightGBMTrainer 等、PyTorch 以外の trainer への適用（今回は Vision/PyTorch 系のみ）
- 既存 `fix_seed()` 関数の削除（後方互換のため残す）
