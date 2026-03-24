"""
VisionInferencer — Vision モデルによる推論実装。

model_dir 配下の fold_N/model.pt を全て読み込み、
各 fold の予測値（class 1 の確率）を平均して返す。

設計上の注意:
- model_dir は fold_N/ サブディレクトリを持つルートディレクトリ。
- model.pt には model_state_dict と backbone_config が保存されている。
- 複数 fold の予測値は単純平均する（アンサンブル戦略は UseCase 層で担う）。

時間計算量: O(F * N * C) — F: fold 数, N: テストサンプル数, C: モデル計算量
空間計算量: O(F * N + P) — 予測結果 + モデルパラメータ
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from src.domain.model.backbone import BackboneConfig
from src.infrastructure.trainer.backbone_registry import build_backbone, build_classifier


class VisionInferencer:
    """Vision モデルで fold ごとに予測し平均値を返す。"""

    MODEL_FILENAME = "model.pt"

    def predict_folds(
        self,
        model_dir: Path,
        test_df: pl.DataFrame,
        feature_cols: list[str],
    ) -> np.ndarray:
        """全 fold のモデルで予測し、fold 間の平均を返す。

        Vision モデルの場合、feature_cols は ["image_path"] を期待する。
        test_df の最初のカラム（image_path）から画像を読み込んで推論する。

        Args:
            model_dir: fold_N/ サブディレクトリを持つモデルルートディレクトリ
            test_df: 予測対象 DataFrame（image_path カラムを持つ）
            feature_cols: ["image_path"] を期待

        Returns:
            shape=(n_test,) の予測値 ndarray（fold 間の平均、class 1 の確率）

        Raises:
            ValueError: fold ディレクトリが存在しない場合

        時間計算量: O(F * N * C) — F: fold数, N: テスト画像数, C: 1 画像の推論
        空間計算量: O(F * N + P)
        """
        fold_dirs = sorted(
            [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")],
            key=lambda d: d.name,
        )
        if not fold_dirs:
            raise ValueError(
                f"fold ディレクトリが見つかりません: {model_dir}\n"
                f"'fold_N/' という名前のサブディレクトリが必要です。"
            )

        image_col = feature_cols[0] if feature_cols else "image_path"
        image_paths: list[str] = test_df[image_col].to_list()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fold_preds: list[np.ndarray] = []

        for fold_dir in fold_dirs:
            model_path = fold_dir / self.MODEL_FILENAME
            if not model_path.exists():
                continue

            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            backbone_cfg = checkpoint["backbone_config"]

            config = BackboneConfig(
                backbone_name=backbone_cfg["backbone_name"],
                num_classes=backbone_cfg["num_classes"],
                pretrained=False,
                image_size=backbone_cfg.get("image_size", 32),
            )
            backbone, num_features = build_backbone(config)
            classifier = build_classifier(num_features, config.num_classes)
            model = nn.Sequential(backbone, nn.Flatten(), classifier).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            image_size = config.image_size
            transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            predictions: list[float] = []
            with torch.no_grad():
                for img_path in image_paths:
                    image = Image.open(img_path).convert("RGB")
                    tensor = transform(image).unsqueeze(0).to(device)
                    output = model(tensor)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                    # binary: class 1 の確率
                    if config.num_classes == 2:
                        predictions.append(float(probs[1]))
                    else:
                        predictions.append(float(probs.max()))

            fold_preds.append(np.array(predictions, dtype=np.float64))

        if not fold_preds:
            raise ValueError(
                f"有効なモデルファイルが見つかりません: {model_dir}\n"
                f"各 fold_N/ に '{self.MODEL_FILENAME}' が必要です。"
            )

        return np.mean(np.stack(fold_preds, axis=0), axis=0).astype(np.float64)
