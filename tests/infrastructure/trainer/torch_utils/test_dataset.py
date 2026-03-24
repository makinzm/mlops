"""
torch_utils/dataset のテスト。

なぜこのテストが必要か:
  - ImageClassificationDataset が画像パスとラベルから正しいテンソルを返すことを確認する。
  - transform が適用されることを確認する。
  - albumentations transform にも対応することを確認する。

時間計算量: O(N * H * W) — N: テスト画像数
空間計算量: O(C * H * W)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.infrastructure.trainer.torch_utils.dataset import ImageClassificationDataset


def _create_images(image_dir: Path, num: int, size: int = 8) -> list[str]:
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    paths: list[str] = []
    for i in range(num):
        data = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        path = image_dir / f"img_{i}.png"
        Image.fromarray(data).save(path)
        paths.append(str(path))
    return paths


class TestImageClassificationDataset:
    def test_len(self, tmp_path: Path) -> None:
        """__len__ がサンプル数を返すこと。"""
        paths = _create_images(tmp_path / "imgs", 5)
        ds = ImageClassificationDataset(paths, [0, 1, 0, 1, 0])
        assert len(ds) == 5

    def test_getitem_returns_tensor_and_label(self, tmp_path: Path) -> None:
        """__getitem__ が (Tensor, int) を返すこと。"""
        paths = _create_images(tmp_path / "imgs", 3)
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        ds = ImageClassificationDataset(paths, [0, 1, 0], torchvision_transform=transform)
        tensor, label = ds[0]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 32, 32)
        assert label == 0

    def test_albumentations_transform(self, tmp_path: Path) -> None:
        """albumentations transform が適用されること。"""
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
        except ImportError:
            return  # albumentations 未インストール時はスキップ

        paths = _create_images(tmp_path / "imgs", 3)
        album_transform = A.Compose([
            A.Resize(32, 32),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        ds = ImageClassificationDataset(paths, [0, 1, 0], albumentations_transform=album_transform)
        tensor, label = ds[0]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 32, 32)
