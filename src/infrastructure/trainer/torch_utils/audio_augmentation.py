"""
音声スペクトログラム用 Data Augmentation。

spec_augment() / mixup() を提供する。

既存の torch_utils/augmentation.py は画像（PIL/albumentations 前提）の
augmentation パイプラインを構築するためのモジュールであり、入力形式が
（H, W, C）の画像であることを前提にしている。
本モジュールはメルスペクトログラム（n_mels, time_frames を持つテンソル）を
直接受け取って変換する点で役割が異なるため、独立したファイルに分離する。

時間計算量: spec_augment は O(B * n_masks), mixup は O(B) — B: バッチサイズ
空間計算量: O(1)（spec_augment は in-place、mixup は入力と同サイズの出力を生成）
"""

from __future__ import annotations

import numpy as np
import torch


def spec_augment(
    mel: torch.Tensor,
    freq_mask_param: int = 20,
    time_mask_param: int = 40,
    n_masks: int = 2,
) -> torch.Tensor:
    """SpecAugment: 周波数帯 + 時間帯をランダムマスクする。

    Args:
        mel: (batch, 1, n_mels, time) テンソル。
        freq_mask_param: 周波数マスクの最大幅。
        time_mask_param: 時間マスクの最大幅。
        n_masks: マスクの個数。

    Returns:
        マスク適用後のテンソル（入力を in-place 変更して返す）。

    時間計算量: O(B * n_masks) — B: バッチサイズ
    空間計算量: O(1)（in-place）
    """
    _, _, n_mels, n_time = mel.shape
    for _ in range(n_masks):
        f = torch.randint(0, freq_mask_param + 1, (1,)).item()
        f0 = torch.randint(0, int(max(n_mels - f, 1)), (1,)).item()
        mel[:, :, f0 : f0 + f, :] = 0.0
        t = torch.randint(0, time_mask_param + 1, (1,)).item()
        t0 = torch.randint(0, int(max(n_time - t, 1)), (1,)).item()
        mel[:, :, :, t0 : t0 + t] = 0.0
    return mel


def mixup(
    mel: torch.Tensor, label: torch.Tensor, alpha: float = 0.4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixup: バッチ内の2サンプルを混合する。multi-label 分類に特に有効。

    Args:
        mel: (batch, 1, n_mels, time)。
        label: (batch, num_classes)。
        alpha: Beta 分布のパラメータ。

    Returns:
        混合後の (mel, label)。

    時間計算量: O(B) — B: バッチサイズ
    空間計算量: O(B)
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = mel.size(0)
    perm = torch.randperm(batch_size, device=mel.device)
    mixed_mel = lam * mel + (1 - lam) * mel[perm]
    mixed_label = lam * label + (1 - lam) * label[perm]
    return mixed_mel, mixed_label
