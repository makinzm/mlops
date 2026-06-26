"""
音声データのドメイン定義。

AudioSample dataclass, AudioLoader Protocol, SpectrogramConfig dataclass,
SpectrogramTransformer Protocol を定義する。
インフラ実装（soundfile, torch 等）はこれらの Protocol を満たすことで
usecase から利用可能になる。

domain 層では numpy/torch 等の外部ライブラリに依存せず、
波形データは list[float]、スペクトログラムは list[list[float]] として保持する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class AudioSample:
    """音声サンプルの不変表現。

    Attributes:
        waveform: 波形データ（-1.0 ~ 1.0 の正規化済み振幅値）。
        sample_rate: サンプリングレート（Hz）。
        duration_seconds: 音声の長さ（秒）。
        file_path: 元の音声ファイルパス。
    """

    waveform: list[float]
    sample_rate: int
    duration_seconds: float
    file_path: str


@runtime_checkable
class AudioLoader(Protocol):
    """音声ファイルを読み込む Protocol。

    時間計算量: O(n) — n: 音声サンプル数
    空間計算量: O(n)
    """

    def load(self, file_path: str) -> AudioSample:
        """音声ファイルを読み込み AudioSample を返す。

        Args:
            file_path: 読み込む音声ファイルのパス。

        Returns:
            AudioSample: 読み込んだ音声データ。
        """
        ...


@dataclass(frozen=True)
class SpectrogramConfig:
    """メルスペクトログラム生成のパラメータ。

    Attributes:
        sample_rate: サンプリングレート（Hz）。
        n_fft: FFT ウィンドウサイズ。
        hop_length: ホップ長。
        n_mels: メルフィルタバンク数。
        segment_seconds: 1セグメントの長さ（秒）。
    """

    sample_rate: int = 32000
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    segment_seconds: float = 5.0

    @property
    def segment_samples(self) -> int:
        """1セグメントのサンプル数。

        時間計算量: O(1)
        空間計算量: O(1)
        """
        return int(self.sample_rate * self.segment_seconds)


@runtime_checkable
class SpectrogramTransformer(Protocol):
    """音声ファイルをスペクトログラムに変換する Protocol。

    返り値の型は実装依存（torch.Tensor 等）。domain 層では型を限定しない。

    時間計算量: O(n * n_fft) — n: 音声サンプル数
    空間計算量: O(n_mels * n_frames)
    """

    def transform_file(self, file_path: str) -> Any:
        """音声ファイルをスペクトログラムに変換する。

        Args:
            file_path: 音声ファイルのパス。

        Returns:
            スペクトログラム表現。型は実装依存（torch.Tensor 等）。
        """
        ...
