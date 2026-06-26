"""
torch + numpy + soundfile によるメルスペクトログラム変換。

torchaudio に依存せず、メルフィルタバンクを numpy で構築し、
torch.stft で STFT を計算してメルスペクトログラム（dB スケール）を生成する。

時間計算量: O(n * n_fft) — n: 音声サンプル数
空間計算量: O(n_mels * n_frames)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.domain.data.audio import SpectrogramConfig


def _create_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> torch.Tensor:
    """メルフィルタバンクを作成する。

    Args:
        sr: サンプリングレート。
        n_fft: FFT サイズ。
        n_mels: メルバンド数。

    Returns:
        (n_mels, n_fft // 2 + 1) の FloatTensor。

    時間計算量: O(n_mels * n_fft)
    空間計算量: O(n_mels * n_fft)
    """
    n_freqs = n_fft // 2 + 1

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_low = hz_to_mel(0)
    mel_high = hz_to_mel(sr / 2)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        for j in range(left, center):
            if center != left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)

    return torch.FloatTensor(filterbank)


class MelSpectrogramTransformer:
    """torch + numpy ベースのメルスペクトログラム変換。

    SpectrogramTransformer Protocol（domain）を満たす。

    Attributes:
        _cfg: スペクトログラム設定。
        _mel_fb: メルフィルタバンク。
        _window: ハニング窓。
    """

    def __init__(self, cfg: SpectrogramConfig) -> None:
        """変換器を初期化する。

        Args:
            cfg: メルスペクトログラムのパラメータ。

        時間計算量: O(n_mels * n_fft)
        空間計算量: O(n_mels * n_fft)
        """
        self._cfg = cfg
        self._mel_fb = _create_mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels)
        self._window = torch.hann_window(cfg.n_fft)

    def transform(self, waveform: torch.Tensor) -> torch.Tensor:
        """波形をメルスペクトログラム（dB）に変換する。

        segment_seconds 未満の音声はゼロパディング、超える音声はトランケートされる。

        Args:
            waveform: 1D テンソル（samples,）。

        Returns:
            2D テンソル（n_mels, time_frames）。

        時間計算量: O(n * n_fft)
        空間計算量: O(n_mels * n_frames)
        """
        target_len = self._cfg.segment_samples
        if waveform.shape[0] < target_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[0]))
        elif waveform.shape[0] > target_len:
            waveform = waveform[:target_len]

        spec = torch.stft(
            waveform,
            n_fft=self._cfg.n_fft,
            hop_length=self._cfg.hop_length,
            window=self._window,
            return_complex=True,
        )
        power = spec.abs() ** 2

        mel = self._mel_fb @ power

        mel_db = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        top_db = 80.0
        mel_db = torch.clamp(mel_db, min=mel_db.max() - top_db)

        return mel_db

    def segment_and_transform(self, waveform: torch.Tensor) -> list[torch.Tensor]:
        """長い音声を segment_seconds 単位に分割してそれぞれ変換する。

        Args:
            waveform: 1D テンソル。

        Returns:
            各セグメントのメルスペクトログラムのリスト。

        時間計算量: O(n * n_fft) — n: 全サンプル数
        空間計算量: O(S * n_mels * n_frames) — S: セグメント数
        """
        seg_len = self._cfg.segment_samples
        total = waveform.shape[0]

        segments = []
        start = 0
        while start < total:
            end = start + seg_len
            chunk = waveform[start:end]
            segments.append(self.transform(chunk))
            start = end

        return segments

    def _load_waveform(self, file_path: str) -> torch.Tensor:
        """音声ファイルから波形テンソルを読み込む。

        サンプリングレートが cfg と異なる場合は線形補間でリサンプルする。

        時間計算量: O(n)
        空間計算量: O(n)
        """
        data, sr = sf.read(file_path, dtype="float32")

        if data.ndim > 1:
            data = data.mean(axis=1)

        waveform = torch.from_numpy(data)

        if sr != self._cfg.sample_rate:
            ratio = self._cfg.sample_rate / sr
            new_len = int(len(waveform) * ratio)
            waveform = torch.nn.functional.interpolate(
                waveform.unsqueeze(0).unsqueeze(0),
                size=new_len,
                mode="linear",
                align_corners=False,
            ).squeeze()

        return waveform

    def transform_file(self, file_path: str | Path) -> torch.Tensor:
        """音声ファイルを読み込んでメルスペクトログラムに変換する。

        Args:
            file_path: 音声ファイルのパス。

        Returns:
            2D テンソル（n_mels, time_frames）。

        時間計算量: O(n * n_fft)
        空間計算量: O(n_mels * n_frames)
        """
        return self.transform(self._load_waveform(str(file_path)))

    def segment_and_transform_file(self, file_path: str | Path) -> list[torch.Tensor]:
        """音声ファイルを segment_seconds 単位に分割してそれぞれ変換する。

        Args:
            file_path: 音声ファイルのパス。

        Returns:
            各セグメントのメルスペクトログラムのリスト。

        時間計算量: O(n * n_fft) — n: 全サンプル数
        空間計算量: O(S * n_mels * n_frames) — S: セグメント数
        """
        return self.segment_and_transform(self._load_waveform(str(file_path)))
