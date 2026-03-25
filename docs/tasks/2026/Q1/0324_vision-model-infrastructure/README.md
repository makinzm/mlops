# CNN/Vision Model Infrastructure 実装

## 目的

現在 LightGBM ベースの表形式データのみ対応のプロジェクトに、画像分類タスク対応の CNN/Vision モデル基盤を追加する。

## 対象モデル

- ViT (vit_b_16, vit_b_32)
- ResNet (resnet18, resnet34, resnet50)
- MobileNet (mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large)
- Simple CNN (カスタム)

## 設計方針

1. 既存の `Trainer` Protocol / `Inferencer` Protocol をそのまま再利用
2. 画像データは前処理パイプラインが `image_path` カラムを含む parquet を出力 → VisionTrainer の Dataset がオンデマンド読み込み
3. backbone_registry で文字列名 → torchvision モデルコンストラクタをマッピング
4. `check_dimensions()` でダミー入力テンソルによる次元検証
5. GradCAM は infrastructure 実装を shared し、学習後自動生成 + 独立 usecase の 2 エントリーポイント

## Phase 一覧

### Phase 1: mille.toml + 依存 + Domain dataclass

- mille.toml: `torch`, `torchvision`, `pytorch_grad_cam`, `PIL` を external_allow に追加
- pyproject.toml: `torchvision>=0.15`, `grad-cam>=1.5` 追加
- `src/domain/model/backbone.py`: `BackboneConfig`, `DimensionInfo` dataclass
- `src/domain/model/gradcam.py`: `GradCAMResult` dataclass + `GradCAMAnalyzer` Protocol

### Phase 2: Backbone Registry + 次元デバッグ

- `src/infrastructure/trainer/backbone_registry.py`: `BACKBONE_REGISTRY`, `build_backbone()`, `build_classifier()`, `check_dimensions()`

### Phase 3: VisionTrainer

- `src/infrastructure/trainer/vision_trainer.py`: `fit_folds()` 実装
- `src/infrastructure/trainer/vision_lightning_module.py`: PyTorch Lightning Module
- `src/main.py`: `_run_train()` に vision 分岐追加

### Phase 4: VisionInferencer

- `src/infrastructure/inference/vision_inferencer.py`: `predict_folds()` 実装
- `src/main.py`: `_run_inference()` に inferencer_type 分岐追加

### Phase 5: GradCAM

- `src/infrastructure/analyzer/gradcam_analyzer.py`: `GradCAMAnalyzerImpl`
- `src/usecase/analysis/gradcam.py`: `GradCAMUseCase`
- `conf/usecase/gradcam.yaml`, `conf/config.yaml` に gradcam キー追加

### Phase 6: 画像前処理 Resolver + ドキュメント

- `src/infrastructure/preprocessor/resolvers/image_resolver.py`: `ImageResolver`
- `docs/manual/vision_training.md`: 手動操作手順書

## 検証方法

各 Phase で:
1. `uv run mille check` — レイヤー境界違反なし
2. `uv run pytest tests/` — 全テスト GREEN
3. `uv run ty check src/ tests/` — 型チェック通過
4. `uv run ruff check src/ tests/` — lint 通過
