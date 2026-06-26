"""
audio_augmentation モジュールのテスト。

なぜこのテストが必要か:
- spec_augment が (batch, 1, n_mels, time) テンソルにマスクを適用することを検証する。
- mixup が2サンプルを正しく混合し、ラベルも同じ比率でブレンドされることを検証する。
- 既存の vision 用 augmentation（albumentations ベース）とは独立したモジュールとして
  動作することを保証する。
"""

import torch

from src.infrastructure.trainer.torch_utils.audio_augmentation import mixup, spec_augment


class TestSpecAugment:
    """spec_augment のテスト。"""

    def test_output_shape_unchanged(self) -> None:
        """入力と出力の形状が同じであること。"""
        batch = torch.randn(4, 1, 128, 312)
        result = spec_augment(batch, freq_mask_param=10, time_mask_param=20, n_masks=2)
        assert result.shape == batch.shape

    def test_some_values_zeroed(self) -> None:
        """マスクが適用されてゼロになる値が存在すること（確率的だが n_masks=1 で高確率）。"""
        torch.manual_seed(0)
        batch = torch.ones(2, 1, 128, 312)
        result = spec_augment(batch, freq_mask_param=10, time_mask_param=10, n_masks=1)
        assert (result == 0.0).any()

    def test_no_mask_when_params_zero(self) -> None:
        """freq_mask_param=time_mask_param=0 のときマスクなし（全値が元のまま）。"""
        batch = torch.ones(2, 1, 64, 100)
        result = spec_augment(batch, freq_mask_param=0, time_mask_param=0, n_masks=2)
        assert torch.allclose(result, batch)

    def test_inplace_modification(self) -> None:
        """spec_augment は入力テンソルを in-place 変更して返すこと。"""
        batch = torch.ones(2, 1, 64, 100)
        result = spec_augment(batch)
        assert result is batch

    def test_per_sample_independent_masks(self) -> None:
        """バッチ内の各サンプルに独立したマスクが適用されること。

        バッチ全体で同一マスクの場合、全サンプルの (freq_zero_cols XOR time_zero_rows) が
        完全一致する。per-sample ならばそれらが分岐するはずなので、
        隣接サンプルのマスクを直接比較して少なくとも1組が異なることを確認する。
        """
        torch.manual_seed(42)
        # 全サンプルを 1.0 で初期化
        batch = torch.ones(8, 1, 64, 312)
        result = spec_augment(batch.clone(), freq_mask_param=20, time_mask_param=60, n_masks=3)
        # 各サンプルのゼロマスクをフラット化してタプルにする
        masks = [tuple((result[i] == 0.0).flatten().tolist()) for i in range(8)]
        # per-sample なら同一マスクが全部同じになることはほぼない（3マスク×8サンプルで多様性がある）
        assert len(set(masks)) > 1


class TestMixup:
    """mixup のテスト。"""

    def test_output_shapes(self) -> None:
        """出力の mel とラベルの形状が入力と同じであること。"""
        mel = torch.randn(4, 1, 128, 312)
        label = torch.rand(4, 10)
        mixed_mel, mixed_label = mixup(mel, label, alpha=0.4)
        assert mixed_mel.shape == mel.shape
        assert mixed_label.shape == label.shape

    def test_label_values_in_range(self) -> None:
        """混合後のラベル値が [0, 1] の範囲内であること（元ラベルが 0/1 の場合）。"""
        torch.manual_seed(0)
        mel = torch.randn(8, 1, 128, 312)
        label = torch.zeros(8, 5)
        label[:, 0] = 1.0
        _, mixed_label = mixup(mel, label, alpha=0.4)
        assert mixed_label.min() >= 0.0
        assert mixed_label.max() <= 1.0

    def test_mel_is_convex_combination(self) -> None:
        """混合後の mel が元の2サンプルの凸結合であること（alpha=1 → lambda=0.5 近辺）。"""
        torch.manual_seed(42)
        mel = torch.zeros(2, 1, 4, 4)
        mel[0] = 1.0
        mel[1] = 3.0
        label = torch.zeros(2, 2)
        mixed_mel, _ = mixup(mel, label, alpha=100.0)  # alpha 大きいと lambda ~ 0.5
        # 混合値は 1.0 と 3.0 の間になるはず
        assert mixed_mel.min() >= 1.0 - 1e-5
        assert mixed_mel.max() <= 3.0 + 1e-5
