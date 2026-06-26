"""
soundfile による音声読み込み。

AudioLoader Protocol の実装。WAV/FLAC/OGG 等の音声ファイルを読み込み、
AudioSample として返す。

時間計算量: O(n) — n: 読み込む音声サンプル数
空間計算量: O(n)
"""

from __future__ import annotations

import soundfile as sf

from src.domain.data.audio import AudioSample


class SoundfileLoader:
    """soundfile ライブラリを使った音声読み込み。

    AudioLoader Protocol を満たす。
    """

    def load(
        self,
        file_path: str,
        offset: float = 0.0,
        duration: float | None = None,
    ) -> AudioSample:
        """音声ファイルを読み込み AudioSample を返す。

        Args:
            file_path: 音声ファイルのパス。
            offset: 読み込み開始位置（秒）。
            duration: 読み込む長さ（秒）。None の場合は全体を読み込む。

        Returns:
            AudioSample: 読み込んだ音声データ。

        時間計算量: O(n) — n: 読み込む音声サンプル数
        空間計算量: O(n)
        """
        info = sf.info(file_path)
        sample_rate = info.samplerate

        start_frame = int(offset * sample_rate)
        frames_to_read = int(duration * sample_rate) if duration is not None else -1

        data, _ = sf.read(
            file_path,
            start=start_frame,
            stop=start_frame + frames_to_read if frames_to_read > 0 else None,
            dtype="float32",
            always_2d=False,
        )

        if data.ndim > 1:
            data = data.mean(axis=1)

        waveform = data.tolist()
        actual_duration = len(waveform) / sample_rate

        return AudioSample(
            waveform=waveform,
            sample_rate=sample_rate,
            duration_seconds=actual_duration,
            file_path=file_path,
        )
