"""
VisionTrainer — Trainer Protocol の Vision モデル実装。

設計:
  fit_folds() が fold ごとに以下を実行する:
    1. preprocess_output_dir/fold_{N}/train.parquet / test.parquet を読み込み
    2. ImageClassificationDataset で画像をオンデマンドに読み込む
    3. backbone_registry で backbone + classifier を構築
    4. PyTorch の訓練ループ（epoch × batch）
    5. fold_{N}/model.pt に保存
    6. validation セットで予測 → oof_train.parquet
    7. error_analysis.parquet（TP/TN/FP/FN サンプリング）
  CV スコアを集計して TrainResult を返す。

  timestamp / commit_hash は TrainUseCase が生成して cfg に含めて渡す。

時間計算量: O(E * N * (C_fwd + C_bwd)) — E: エポック数, N: サンプル数, C: モデル計算量
空間計算量: O(P + B * C * H * W) — P: パラメータ数, B: バッチサイズ, C*H*W: 画像サイズ
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.domain.model.backbone import BackboneConfig
from src.domain.model.trainer import FoldResult, TrainResult
from src.infrastructure.trainer.backbone_registry import build_backbone, build_classifier

logger = logging.getLogger(__name__)


class ImageClassificationDataset(Dataset):
    """画像分類用 PyTorch Dataset。

    image_path カラムから画像を読み込み、transform を適用する。

    時間計算量: __getitem__ は O(H * W) — 画像読み込み + transform
    空間計算量: O(C * H * W) — 1 画像分
    """

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        transform: transforms.Compose | None = None,
    ) -> None:
        self._image_paths = image_paths
        self._labels = labels
        self._transform = transform or transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:  # ty:ignore[invalid-method-override]
        image = Image.open(self._image_paths[idx]).convert("RGB")
        tensor = self._transform(image)
        return tensor, self._labels[idx]


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

        時間計算量: O(F * E * N * C) — F: fold数, E: エポック, N: サンプル, C: モデル計算
        空間計算量: O(P + B * C * H * W)
        """
        dcfg = self._cfg
        fold_dirs = sorted(preprocess_output_dir.glob("fold_*"))
        if not fold_dirs:
            raise ValueError(f"fold ディレクトリが見つかりません: {preprocess_output_dir}")

        timestamp: str = cfg.get("_timestamp", datetime.now().strftime("%Y%m%dT%H%M%S"))
        commit_hash: str = cfg.get("_commit_hash", "unknown")
        job_id: str = cfg.get("job_id", "vision")
        target_col: str = dcfg["target_col"]
        image_path_col: str = dcfg.get("image_path_col", "image_path")
        seed: int = int(dcfg["seed"])
        num_classes: int = int(dcfg["num_classes"])
        n_error: int = int(dcfg.get("report", {}).get("n_error_samples", 50))

        # backbone 設定
        backbone_cfg = dcfg.get("backbone", {})
        backbone_name: str = backbone_cfg.get("name", "simple_cnn")
        pretrained: bool = backbone_cfg.get("pretrained", False)
        image_size: int = int(backbone_cfg.get("image_size", 32))

        # 学習設定
        training_cfg = dcfg.get("training", {})
        num_epochs: int = int(training_cfg.get("num_epochs", 10))
        batch_size: int = int(training_cfg.get("batch_size", 32))
        learning_rate: float = float(training_cfg.get("learning_rate", 0.001))
        num_workers: int = int(training_cfg.get("num_workers", 0))

        backbone_config = BackboneConfig(
            backbone_name=backbone_name,
            num_classes=num_classes,
            pretrained=pretrained,
            image_size=image_size,
        )

        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fold_results: list[FoldResult] = []

        for fold_dir in fold_dirs:
            fold_idx = int(fold_dir.name.replace("fold_", ""))
            fold_out = output_dir / f"fold_{fold_idx}"
            fold_out.mkdir(parents=True, exist_ok=True)

            # seed 固定（再現性）
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(seed)

            train_df = pl.read_parquet(fold_dir / "train.parquet")
            valid_df = pl.read_parquet(fold_dir / "test.parquet")

            train_paths = train_df[image_path_col].to_list()
            train_labels = train_df[target_col].to_list()
            valid_paths = valid_df[image_path_col].to_list()
            valid_labels = valid_df[target_col].to_list()

            train_dataset = ImageClassificationDataset(train_paths, train_labels, transform)
            valid_dataset = ImageClassificationDataset(valid_paths, valid_labels, transform)

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

            # モデル構築
            backbone, num_features = build_backbone(backbone_config)
            classifier = build_classifier(num_features, num_classes)
            model = nn.Sequential(backbone, nn.Flatten(), classifier).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # 学習ループ
            best_valid_acc = 0.0
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    train_correct += predicted.eq(labels).sum().item()
                    train_total += labels.size(0)

                # Validation
                model.eval()
                valid_correct = 0
                valid_total = 0
                with torch.no_grad():
                    for images, labels in valid_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, predicted = outputs.max(1)
                        valid_correct += predicted.eq(labels).sum().item()
                        valid_total += labels.size(0)

                valid_acc = valid_correct / max(valid_total, 1)
                if valid_acc >= best_valid_acc:
                    best_valid_acc = valid_acc
                    # ベストモデル保存
                    model_path = fold_out / "model.pt"
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "backbone_config": {
                                "backbone_name": backbone_name,
                                "num_classes": num_classes,
                                "pretrained": pretrained,
                                "image_size": image_size,
                            },
                        },
                        model_path,
                    )

                train_acc = train_correct / max(train_total, 1)
                logger.info(
                    f"Fold {fold_idx} Epoch {epoch + 1}/{num_epochs}: "
                    f"train_acc={train_acc:.4f}, valid_acc={valid_acc:.4f}"
                )

            # OOF 予測保存
            model.eval()
            all_probs: list[np.ndarray] = []
            with torch.no_grad():
                for images, _ in valid_loader:
                    images = images.to(device)
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    all_probs.append(probs)

            valid_probs = np.concatenate(all_probs, axis=0)
            # binary の場合は class 1 の確率
            if num_classes == 2:
                predicted_proba = valid_probs[:, 1]
            else:
                predicted_proba = valid_probs.max(axis=1)

            oof_df = valid_df.select([target_col, image_path_col]).to_pandas()
            oof_df["predicted_proba"] = predicted_proba
            oof_path = fold_out / "oof_train.parquet"
            pl.from_pandas(oof_df).write_parquet(oof_path)

            # error_analysis 生成
            ea_path = fold_out / "error_analysis.parquet"
            _write_error_analysis(
                valid_df.to_pandas(),
                predicted_proba,
                valid_labels,
                target_col,
                image_path_col,
                n_error,
                ea_path,
            )

            model_path = fold_out / "model.pt"
            fold_results.append(
                FoldResult(
                    fold_idx=fold_idx,
                    train_score=train_acc,
                    valid_score=best_valid_acc,
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
        cv_mean = float(np.mean(scores))
        cv_std = float(np.std(scores))

        return TrainResult(
            job_id=job_id,
            timestamp=timestamp,
            commit_hash=commit_hash,
            trainer_type="vision",
            output_dir=output_dir,
            fold_results=fold_results,
            cv_mean_score=cv_mean,
            cv_std_score=cv_std,
            metric="accuracy",
            seed=seed,
        )


def _write_error_analysis(
    df: object,
    preds: np.ndarray,
    labels: list[int],
    target_col: str,
    image_path_col: str,
    n_samples: int,
    out_path: Path,
) -> None:
    """4 種サンプリング（TP/TN/FP/FN）を out_path に保存する。

    時間計算量: O(N log N) — ソートあり
    空間計算量: O(N)
    """
    import pandas as pd

    y = np.array(labels)
    threshold = 0.5
    pred_label = (preds >= threshold).astype(int)

    result = pd.DataFrame(
        {
            image_path_col: pd.DataFrame(df)[image_path_col].values,
            "target": y,
            "predicted_proba": preds,
            "predicted_label": pred_label,
            "is_correct": (pred_label == y).astype(int),
            "error_magnitude": np.abs(preds - y),
        }
    )

    def _label(t: int, p: int) -> str:
        if t == 1 and p == 1:
            return "TP"
        if t == 0 and p == 0:
            return "TN"
        if t == 0 and p == 1:
            return "FP"
        return "FN"

    result["sample_type"] = [_label(int(t), int(p)) for t, p in zip(y, pred_label)]

    samples: list[pd.DataFrame] = []
    for stype in ("TP", "TN", "FP", "FN"):
        subset = result[result["sample_type"] == stype]
        if stype in ("FP", "FN"):
            subset = subset.sort_values("error_magnitude", ascending=False)
        else:
            subset = subset.sort_values("error_magnitude", ascending=True)
        samples.append(subset.head(n_samples))

    combined = pd.concat(samples, ignore_index=True)
    pl.from_pandas(combined).write_parquet(out_path)
