# TODO: Vision Model Infrastructure 拡張

## Batch 1: torch_utils/ 抽出
- [ ] `src/infrastructure/trainer/torch_utils/__init__.py`
- [ ] `src/infrastructure/trainer/torch_utils/seed.py` — fix_seed()
- [ ] `src/infrastructure/trainer/torch_utils/dataset.py` — ImageClassificationDataset
- [ ] `src/infrastructure/trainer/torch_utils/training_loop.py` — run_training_loop()
- [ ] `src/infrastructure/trainer/torch_utils/model_builder.py` — build_model(), load_checkpoint()
- [ ] `src/infrastructure/trainer/error_analysis.py` — LightGBM/Vision 共通化
- [ ] vision_trainer.py, vision_inferencer.py, gradcam_analyzer.py をリファクタ
- [ ] lgbm_trainer.py の error_analysis を共通モジュールに差し替え

## Batch 2: 入力バリデーション
- [ ] `src/infrastructure/trainer/torch_utils/validation.py` — validate_training_inputs()
- [ ] VisionTrainer.fit_folds() の先頭で呼び出し
- [ ] 画像サイズ・パス存在・ラベル範囲・次元ミスマッチの検証

## Batch 3: Config-driven Custom CNN
- [ ] Domain: ConvBlockConfig, SkipConnectionConfig, CustomCNNConfig
- [ ] `src/infrastructure/trainer/custom_cnn.py` — ResidualBlock, InvertedBottleneckBlock, CustomCNNModule
- [ ] backbone_registry に register_backbone() + custom_cnn ビルダー追加

## Batch 4: Data Augmentation
- [ ] Domain: AugmentTransformConfig, AugmentationConfig
- [ ] `src/infrastructure/trainer/torch_utils/augmentation.py` — build_augmentation_pipeline()
- [ ] Dataset に albumentations 統合
- [ ] pyproject.toml に albumentations 追加

## Batch 5: Image Preprocessing Resolver
- [ ] `src/infrastructure/preprocessor/resolvers/image_resolver.py`
- [ ] registry.py に登録

## Batch 6: Remote Train パス修正
- [ ] _PROJECT_ROOT を動的に解決
- [ ] Vision trainer 用の deps を bootstrap に追加

## Batch 7: マニュアル
- [ ] docs/manual/vision_training.md 全面更新
