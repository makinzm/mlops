"""
GradCAMAnalyzerImpl — GradCAM 分析の infrastructure 実装。

pytorch_grad_cam ライブラリを使って GradCAM ヒートマップを生成する。
backbone の種類に応じてターゲットレイヤーを自動検出する。

設計:
  - CNN 系 → 最終 Conv2d レイヤー
  - ViT 系 → 最終 attention block の LayerNorm
  - SimpleCNN → features 内の最終 Conv2d

時間計算量: O(N * P) — N: 画像数, P: モデルのフォワード+バックワードパス計算量
空間計算量: O(P + H * W) — モデルパラメータ + ヒートマップサイズ
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM  # ty:ignore[unresolved-import]
from pytorch_grad_cam.utils.image import show_cam_on_image  # ty:ignore[unresolved-import]
from torch import nn
from torchvision import transforms

from src.domain.model.backbone import BackboneConfig
from src.domain.model.gradcam import GradCAMResult
from src.infrastructure.trainer.backbone_registry import build_backbone, build_classifier

logger = logging.getLogger(__name__)


def _find_target_layer(model: nn.Module, backbone_name: str) -> list[nn.Module]:
    """backbone 名に応じてGradCAM のターゲットレイヤーを自動検出する。

    時間計算量: O(L) — L: モデルレイヤー数
    空間計算量: O(1)
    """
    if backbone_name.startswith("vit"):
        # ViT: encoder の最終ブロックの LayerNorm
        encoder = model[0].encoder  # ty:ignore[not-subscriptable]
        return [encoder.layers[-1].ln_1]  # ty:ignore[not-subscriptable]

    # CNN 系: 最終 Conv2d レイヤーを探す
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        return [last_conv]

    raise ValueError(f"ターゲットレイヤーが見つかりません: {backbone_name}")


class GradCAMAnalyzerImpl:
    """GradCAM 分析の実装。GradCAMAnalyzer Protocol を満たす。"""

    def analyze(
        self,
        model_path: Path,
        image_paths: list[Path],
        output_dir: Path,
        target_layer: str | None = None,
    ) -> list[GradCAMResult]:
        """指定モデルで画像の GradCAM ヒートマップを生成する。

        Args:
            model_path: 学習済みモデルファイルのパス（.pt）
            image_paths: 分析対象の画像パスリスト
            output_dir: ヒートマップの出力先ディレクトリ
            target_layer: GradCAM のターゲットレイヤー名（None の場合自動検出）

        Returns:
            各画像の GradCAMResult リスト

        時間計算量: O(N * (C_fwd + C_bwd)) — N: 画像数
        空間計算量: O(P + H * W)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        target_layers = _find_target_layer(model, config.backbone_name)

        results: list[GradCAMResult] = []
        cam = GradCAM(model=model, target_layers=target_layers)

        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)  # ty:ignore[unresolved-attribute]

            # GradCAM 生成
            grayscale_cam = cam(input_tensor=input_tensor)
            grayscale_cam = grayscale_cam[0, :]

            # 予測
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                predicted_class = int(probs.argmax())
                confidence = float(probs.max())

            # ヒートマップ画像保存
            rgb_image = np.array(image.resize((image_size, image_size))).astype(np.float32) / 255.0
            cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
            heatmap_path = output_dir / f"gradcam_{img_path.stem}.png"
            Image.fromarray(cam_image).save(heatmap_path)

            results.append(
                GradCAMResult(
                    image_path=img_path,
                    heatmap_path=heatmap_path,
                    predicted_class=predicted_class,
                    confidence=confidence,
                )
            )

        return results
