"""
画像分類用 PyTorch Dataset。

torchvision.transforms と albumentations の両方に対応する。
albumentations が指定された場合は PIL → numpy 変換を行ってから適用する。

時間計算量: __getitem__ は O(H * W) — 画像読み込み + transform
空間計算量: O(C * H * W) — 1 画像分
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageClassificationDataset(Dataset):
    """画像分類用 PyTorch Dataset。

    torchvision_transform か albumentations_transform のどちらかを指定する。
    両方指定された場合は albumentations を優先する。
    どちらも未指定の場合は Resize(32) + ToTensor() + Normalize をデフォルトで使う。
    """

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        torchvision_transform: transforms.Compose | None = None,
        albumentations_transform: Any | None = None,
    ) -> None:
        self._image_paths = image_paths
        self._labels = labels
        self._album_transform = albumentations_transform
        self._tv_transform = torchvision_transform or transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:  # ty:ignore[invalid-method-override]
        """画像を読み込み、transform を適用して (tensor, label) を返す。

        時間計算量: O(H * W)
        空間計算量: O(C * H * W)
        """
        image = Image.open(self._image_paths[idx]).convert("RGB")

        if self._album_transform is not None:
            image_np = np.array(image)
            augmented = self._album_transform(image=image_np)
            tensor = augmented["image"]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.from_numpy(tensor).permute(2, 0, 1).float()
        else:
            tensor = self._tv_transform(image)

        return tensor, self._labels[idx]
