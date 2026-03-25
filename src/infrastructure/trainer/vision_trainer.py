"""
VisionTrainer — Trainer Protocol の Vision モデル実装。

torch_utils/ のユーティリティを使って fold ごとの学習を実行する。
  1. 入力バリデーション（validate_training_inputs）
  2. seed 固定（fix_seed）
  3. Dataset 構築（ImageClassificationDataset）
  4. モデル構築（build_vision_model）
  5. 学習ループ（run_training_loop）
  6. チェックポイント保存（save_checkpoint）
  7. OOF 予測 + error_analysis（write_error_analysis）

timestamp / commit_hash は TrainUseCase が生成して cfg に含めて渡す。

時間計算量: O(F * E * N * C) — F: fold, E: epoch, N: sample, C: model computation
空間計算量: O(P + B * C * H * W)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.domain.model.backbone import BackboneConfig
from src.domain.model.trainer import FoldResult, TrainResult
from src.infrastructure.trainer.error_analysis import write_error_analysis
from src.infrastructure.trainer.torch_utils.augmentation import (
    build_augmentation_pipeline,
    build_default_transform,
)
from src.infrastructure.trainer.torch_utils.dataset import ImageClassificationDataset
from src.infrastructure.trainer.torch_utils.model_builder import (
    build_vision_model,
    save_checkpoint,
)
from src.infrastructure.trainer.torch_utils.seed import fix_seed
from src.infrastructure.trainer.torch_utils.training_loop import run_training_loop
from src.infrastructure.trainer.torch_utils.validation import validate_training_inputs

logger = logging.getLogger(__name__)


class VisionTrainer:
    """Vision モデルによる k-fold クロスバリデーション学習器。

    Trainer Protocol を満たす。
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg

    def fit_folds(
        self,
        preprocess_output_dir: Path,
        output_dir: Path,
        cfg: dict[str, Any],
    ) -> TrainResult:
        """fold ごとに学習し TrainResult を返す。

        時間計算量: O(F * E * N * C)
        空間計算量: O(P + B * C * H * W)
        """
        dcfg = self._cfg
        timestamp: str = cfg.get("_timestamp", datetime.now().strftime("%Y%m%dT%H%M%S"))
        commit_hash: str = cfg.get("_commit_hash", "unknown")
        job_id: str = cfg.get("job_id", "vision")
        target_col: str = dcfg["target_col"]
        image_path_col: str = dcfg.get("image_path_col", "image_path")
        seed: int = int(dcfg["seed"])
        num_classes: int = int(dcfg["num_classes"])
        n_error: int = int(dcfg.get("report", {}).get("n_error_samples", 50))

        backbone_cfg = dcfg.get("backbone", {})
        backbone_name: str = backbone_cfg.get("name", "simple_cnn")
        pretrained: bool = backbone_cfg.get("pretrained", False)
        image_size: int = int(backbone_cfg.get("image_size", 32))

        training_cfg = dcfg.get("training", {})
        num_epochs: int = int(training_cfg.get("num_epochs", 10))
        batch_size: int = int(training_cfg.get("batch_size", 32))
        learning_rate: float = float(training_cfg.get("learning_rate", 0.001))
        num_workers: int = int(training_cfg.get("num_workers", 0))

        # CustomCNNConfig の構築
        custom_cnn_config = None
        if backbone_name == "custom_cnn" and backbone_cfg.get("custom_cnn"):
            from src.domain.model.custom_cnn import (
                ConvBlockConfig,
                CustomCNNConfig,
                SkipConnectionConfig,
            )

            raw_cnn = backbone_cfg["custom_cnn"]
            layers = [ConvBlockConfig(**layer) for layer in raw_cnn.get("layers", [])]
            skip_connections = None
            if raw_cnn.get("skip_connections"):
                skip_connections = [
                    SkipConnectionConfig(**sc) for sc in raw_cnn["skip_connections"]
                ]
            custom_cnn_config = CustomCNNConfig(
                layers=layers,
                skip_connections=skip_connections,
                adaptive_pool_size=raw_cnn.get("adaptive_pool_size", 1),
            )

        backbone_config = BackboneConfig(
            backbone_name=backbone_name,
            num_classes=num_classes,
            pretrained=pretrained,
            image_size=image_size,
            custom_cnn_config=custom_cnn_config,
        )

        # 入力バリデーション
        issues = validate_training_inputs(
            preprocess_output_dir=preprocess_output_dir,
            backbone_config=backbone_config,
            target_col=target_col,
            image_path_col=image_path_col,
            num_classes=num_classes,
        )
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            msg = "\n".join(f"[{e.field}] {e.message}" for e in errors)
            raise ValueError(f"入力バリデーションエラー:\n{msg}")
        for w in [i for i in issues if i.severity == "warning"]:
            logger.warning(f"[{w.field}] {w.message}")

        # Augmentation
        use_albumentations = False
        aug_cfg = dcfg.get("augmentation")
        if aug_cfg:
            from src.domain.model.augmentation import AugmentationConfig, AugmentTransformConfig

            train_transforms = [AugmentTransformConfig(**t) for t in aug_cfg.get("train", [])]
            valid_transforms = [AugmentTransformConfig(**t) for t in aug_cfg.get("valid", [])]
            aug_config = AugmentationConfig(
                train_transforms=train_transforms,
                valid_transforms=valid_transforms,
            )
            train_transform, valid_transform = build_augmentation_pipeline(aug_config, image_size)
            # albumentations が使われたかを判定（torchvision.transforms.Compose でなければ album）
            use_albumentations = not isinstance(train_transform, transforms.Compose)
        else:
            train_transform = build_default_transform(image_size)
            valid_transform = build_default_transform(image_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fold_dirs = sorted(preprocess_output_dir.glob("fold_*"))
        fold_results: list[FoldResult] = []

        for fold_dir in fold_dirs:
            fold_idx = int(fold_dir.name.replace("fold_", ""))
            fold_out = output_dir / f"fold_{fold_idx}"
            fold_out.mkdir(parents=True, exist_ok=True)

            fix_seed(seed)

            train_df = pl.read_parquet(fold_dir / "train.parquet")
            valid_df = pl.read_parquet(fold_dir / "test.parquet")

            train_paths = train_df[image_path_col].to_list()
            train_labels = train_df[target_col].to_list()
            valid_paths = valid_df[image_path_col].to_list()
            valid_labels = valid_df[target_col].to_list()

            if use_albumentations:
                train_dataset = ImageClassificationDataset(
                    train_paths, train_labels, albumentations_transform=train_transform
                )
                valid_dataset = ImageClassificationDataset(
                    valid_paths, valid_labels, albumentations_transform=valid_transform
                )
            else:
                train_dataset = ImageClassificationDataset(
                    train_paths, train_labels, torchvision_transform=train_transform
                )
                valid_dataset = ImageClassificationDataset(
                    valid_paths, valid_labels, torchvision_transform=valid_transform
                )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                generator=torch.Generator().manual_seed(seed),
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            model = build_vision_model(backbone_config).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            metrics = run_training_loop(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=num_epochs,
                device=device,
            )

            # ベストモデル保存
            model.load_state_dict(metrics.best_model_state_dict)
            model_path = fold_out / "model.pt"
            save_checkpoint(model, backbone_config, model_path)

            # OOF 予測
            model.eval()
            all_probs: list[np.ndarray] = []
            with torch.no_grad():
                for images, _ in valid_loader:
                    images = images.to(device)
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    all_probs.append(probs)

            valid_probs = np.concatenate(all_probs, axis=0)
            predicted_proba = valid_probs[:, 1] if num_classes == 2 else valid_probs.max(axis=1)

            oof_df = valid_df.select([target_col, image_path_col]).to_pandas()
            oof_df["predicted_proba"] = predicted_proba
            oof_path = fold_out / "oof_train.parquet"
            pl.from_pandas(oof_df).write_parquet(oof_path)

            # error_analysis
            ea_path = fold_out / "error_analysis.parquet"
            write_error_analysis(
                predictions=predicted_proba,
                labels=np.array(valid_labels),
                n_samples=n_error,
                output_path=ea_path,
                extra_columns={image_path_col: valid_paths},
            )

            fold_results.append(
                FoldResult(
                    fold_idx=fold_idx,
                    train_score=metrics.final_train_accuracy,
                    valid_score=metrics.best_valid_accuracy,
                    metric="accuracy",
                    model_path=model_path,
                    oof_path=oof_path,
                    error_analysis_path=ea_path,
                    feature_importance_path=None,
                    n_train=len(train_paths),
                    n_valid=len(valid_paths),
                )
            )

        scores = [f.valid_score for f in fold_results]
        return TrainResult(
            job_id=job_id,
            timestamp=timestamp,
            commit_hash=commit_hash,
            trainer_type="vision",
            output_dir=output_dir,
            fold_results=fold_results,
            cv_mean_score=float(np.mean(scores)),
            cv_std_score=float(np.std(scores)),
            metric="accuracy",
            seed=seed,
        )
