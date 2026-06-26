"""
SoundfileLoader のテスト。

なぜこのテストが必要か:
- soundfile ライブラリを使った音声読み込みが AudioLoader Protocol を満たすことを検証する。
- 読み込んだ AudioSample の waveform, sample_rate, duration_seconds が正しいことを確認する。
- 存在しないファイルの場合に適切なエラーが発生することを保証する。
- offset/duration による部分読み込みが正しく動作することを保証する。
"""

import numpy as np
import pytest
import soundfile as sf

from src.domain.data.audio import AudioLoader, AudioSample
from src.infrastructure.audio.soundfile_loader import SoundfileLoader


@pytest.fixture()
def wav_file(tmp_path):
    """テスト用 WAV ファイルを作成する。"""
    sample_rate = 32000
    duration = 0.1  # 100ms
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440Hz sine
    path = tmp_path / "test.wav"
    sf.write(str(path), waveform, sample_rate)
    return str(path), sample_rate, duration, len(waveform)


class TestSoundfileLoader:
    """SoundfileLoader のテスト。"""

    def test_implements_audio_loader_protocol(self) -> None:
        """AudioLoader Protocol を満たすこと。"""
        loader = SoundfileLoader()
        assert isinstance(loader, AudioLoader)

    def test_load_returns_audio_sample(self, wav_file) -> None:
        """音声ファイルを読み込んで AudioSample を返すこと。"""
        path, _, _, _ = wav_file
        loader = SoundfileLoader()
        result = loader.load(path)
        assert isinstance(result, AudioSample)

    def test_load_correct_sample_rate(self, wav_file) -> None:
        """サンプルレートが正しいこと。"""
        path, sample_rate, _, _ = wav_file
        loader = SoundfileLoader()
        result = loader.load(path)
        assert result.sample_rate == sample_rate

    def test_load_correct_waveform_length(self, wav_file) -> None:
        """波形の長さが正しいこと。"""
        path, _, _, num_samples = wav_file
        loader = SoundfileLoader()
        result = loader.load(path)
        assert len(result.waveform) == num_samples

    def test_load_correct_duration(self, wav_file) -> None:
        """duration_seconds が概ね正しいこと。"""
        path, _, duration, _ = wav_file
        loader = SoundfileLoader()
        result = loader.load(path)
        assert abs(result.duration_seconds - duration) < 0.01

    def test_load_correct_file_path(self, wav_file) -> None:
        """file_path が正しく設定されること。"""
        path, _, _, _ = wav_file
        loader = SoundfileLoader()
        result = loader.load(path)
        assert result.file_path == path

    def test_load_nonexistent_file_raises(self) -> None:
        """存在しないファイルで例外が発生すること。"""
        loader = SoundfileLoader()
        with pytest.raises(Exception):
            loader.load("/nonexistent/file.wav")

    def test_load_with_offset_and_duration(self, wav_file) -> None:
        """オフセットと期間を指定して部分読み込みできること。"""
        path, sample_rate, _, _ = wav_file
        loader = SoundfileLoader()
        result = loader.load(path, offset=0.0, duration=0.05)
        expected_samples = int(sample_rate * 0.05)
        assert len(result.waveform) == expected_samples
