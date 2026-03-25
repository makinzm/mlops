"""
Data Augmentation パイプライン構築。

Albumentations を使った config-driven な augmentation パイプラインを提供する。
albumentations 未インストール時は torchvision にフォールバックする。

時間計算量: パイプライン構築は O(T), 適用は O(H * W * T) — T: transform 数
空間計算量: O(C * H * W)
"""

from __future__ import annotations

import logging
from typing import Any

from torchvision import transforms

from src.domain.model.augmentation import AugmentationConfig, AugmentTransformConfig

logger = logging.getLogger(__name__)


def build_default_transform(image_size: int) -> transforms.Compose:
    """デフォルトの torchvision transform（Resize + ToTensor + Normalize）を返す。

    Args:
        image_size: リサイズ先のサイズ

    Returns:
        transforms.Compose

    時間計算量: O(1)
    空間計算量: O(1)
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_augmentation_pipeline(
    config: AugmentationConfig,
    image_size: int,
) -> tuple[Any, Any]:
    """AugmentationConfig から train/valid の augmentation パイプラインを構築する。

    albumentations がインストールされている場合は albumentations を使い、
    未インストールの場合は torchvision にフォールバックする。

    Args:
        config: augmentation 設定
        image_size: リサイズ先のサイズ

    Returns:
        (train_transform, valid_transform) のタプル。
        albumentations の場合は albumentations.Compose、
        torchvision の場合は transforms.Compose。

    時間計算量: O(T)
    空間計算量: O(1)
    """
    valid_transform = build_default_transform(image_size)

    if not config.train_transforms:
        return valid_transform, valid_transform

    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        train_transform = _build_albumentations_pipeline(config.train_transforms, image_size)
        if config.valid_transforms:
            valid_album = _build_albumentations_pipeline(config.valid_transforms, image_size)
        else:
            valid_album = A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ty:ignore[invalid-argument-type]
                    ToTensorV2(),
                ]
            )
        return train_transform, valid_album

    except ImportError:
        logger.warning("albumentations が未インストールです。torchvision にフォールバックします。")
        return valid_transform, valid_transform


def _build_albumentations_pipeline(
    transforms_config: list[AugmentTransformConfig],
    image_size: int,
) -> Any:
    """AugmentTransformConfig リストから albumentations.Compose を構築する。

    時間計算量: O(T) — T: transform 数
    空間計算量: O(1)
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    album_transforms: list[Any] = [A.Resize(image_size, image_size)]

    for tc in transforms_config:
        transform_cls = getattr(A, tc.name, None)
        if transform_cls is None:
            logger.warning(f"Unknown albumentations transform: {tc.name}, skipping")
            continue
        album_transforms.append(transform_cls(p=tc.probability, **tc.params))

    album_transforms.extend(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ty:ignore[invalid-argument-type]
            ToTensorV2(),
        ]
    )

    return A.Compose(album_transforms)
