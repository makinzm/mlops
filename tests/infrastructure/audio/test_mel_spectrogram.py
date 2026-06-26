"""
MelSpectrogramTransformer のテスト。

なぜこのテストが必要か:
- torchaudio を使わずに numpy + torch で実装したメルスペクトログラムが
  SpectrogramTransformer Protocol を満たすことを検証する。
- 出力形状 (n_mels, time_frames) が設定値と整合することを確認する。
- 短い音声のゼロパディング・長い音声のトランケートが正しく動くことを保証する。
- segment_and_transform が複数セグメントに分割することを保証する。
"""

import numpy as np
import pytest
import soundfile as sf
import torch

from src.domain.data.audio import SpectrogramConfig, SpectrogramTransformer
from src.infrastructure.audio.mel_spectrogram import MelSpectrogramTransformer


@pytest.fixture()
def wav_file(tmp_path):
    """5秒のサイン波 WAV ファイルを生成する。"""
    sample_rate = 32000
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    path = tmp_path / "test.wav"
    sf.write(str(path), waveform, sample_rate)
    return str(path), sample_rate


@pytest.fixture()
def default_cfg() -> SpectrogramConfig:
    return SpectrogramConfig(
        sample_rate=32000, n_fft=512, hop_length=128, n_mels=64, segment_seconds=5.0
    )


class TestMelSpectrogramTransformer:
    def test_implements_spectrogram_transformer_protocol(
        self, default_cfg: SpectrogramConfig
    ) -> None:
        """SpectrogramTransformer Protocol を満たすこと。"""
        transformer = MelSpectrogramTransformer(default_cfg)
        assert isinstance(transformer, SpectrogramTransformer)

    def test_transform_output_shape(self, default_cfg: SpectrogramConfig) -> None:
        """transform の出力形状が (n_mels, *) であること。"""
        transformer = MelSpectrogramTransformer(default_cfg)
        waveform = torch.randn(default_cfg.segment_samples)
        result = transformer.transform(waveform)
        assert result.shape[0] == default_cfg.n_mels
        assert result.dim() == 2

    def test_short_audio_zero_padded(self, default_cfg: SpectrogramConfig) -> None:
        """segment_seconds より短い音声はゼロパディングされること。"""
        transformer = MelSpectrogramTransformer(default_cfg)
        short_waveform = torch.randn(1000)  # segment_samples=160000 より短い
        result = transformer.transform(short_waveform)
        assert result.shape[0] == default_cfg.n_mels

    def test_long_audio_truncated(self, default_cfg: SpectrogramConfig) -> None:
        """segment_seconds より長い音声はトランケートされること。"""
        transformer = MelSpectrogramTransformer(default_cfg)
        long_waveform = torch.randn(default_cfg.segment_samples * 2)
        result = transformer.transform(long_waveform)
        assert result.shape[0] == default_cfg.n_mels

    def test_transform_file(self, wav_file, default_cfg: SpectrogramConfig) -> None:
        """ファイルからメルスペクトログラムを生成できること。"""
        path, _ = wav_file
        transformer = MelSpectrogramTransformer(default_cfg)
        result = transformer.transform_file(path)
        assert result.shape[0] == default_cfg.n_mels

    def test_segment_and_transform_splits_audio(
        self, default_cfg: SpectrogramConfig
    ) -> None:
        """segment_and_transform が複数セグメントを返すこと。"""
        transformer = MelSpectrogramTransformer(default_cfg)
        # 3セグメント分の音声
        waveform = torch.randn(default_cfg.segment_samples * 3)
        segments = transformer.segment_and_transform(waveform)
        assert len(segments) == 3

    def test_output_is_db_scale(self, default_cfg: SpectrogramConfig) -> None:
        """出力が dB スケール（有限な値）であること。"""
        transformer = MelSpectrogramTransformer(default_cfg)
        waveform = torch.randn(default_cfg.segment_samples)
        result = transformer.transform(waveform)
        assert torch.isfinite(result).all()
