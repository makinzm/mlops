"""
AudioSample / AudioLoader / SpectrogramConfig / SpectrogramTransformer のテスト。

なぜこのテストが必要か:
- AudioSample が音声データの不変表現として正しくインスタンス化できることを検証する。
- AudioLoader Protocol を満たす実装が isinstance チェックを通ることを検証する。
- domain 層が numpy 等の外部ライブラリに依存せず、list[float] で波形を保持することを保証する。
- SpectrogramConfig のデフォルト値・segment_samples の計算が正しいことを検証する。
- SpectrogramTransformer Protocol を満たす実装が isinstance チェックを通ることを検証する。
"""

from src.domain.data.audio import (
    AudioLoader,
    AudioSample,
    SpectrogramConfig,
    SpectrogramTransformer,
)


class TestAudioSample:
    """AudioSample dataclass のテスト。"""

    def test_create_audio_sample(self) -> None:
        """基本的なインスタンス生成ができること。"""
        sample = AudioSample(
            waveform=[0.0, 0.5, -0.5, 1.0],
            sample_rate=32000,
            duration_seconds=0.000125,
            file_path="/path/to/audio.wav",
        )
        assert sample.waveform == [0.0, 0.5, -0.5, 1.0]
        assert sample.sample_rate == 32000
        assert sample.duration_seconds == 0.000125
        assert sample.file_path == "/path/to/audio.wav"

    def test_empty_waveform(self) -> None:
        """空の波形でもインスタンス生成できること。"""
        sample = AudioSample(
            waveform=[],
            sample_rate=32000,
            duration_seconds=0.0,
            file_path="/path/to/empty.wav",
        )
        assert sample.waveform == []
        assert sample.duration_seconds == 0.0


class TestAudioLoaderProtocol:
    """AudioLoader Protocol のテスト。"""

    def test_protocol_is_runtime_checkable(self) -> None:
        """AudioLoader が runtime_checkable であること。"""

        class _MockLoader:
            def load(self, file_path: str) -> AudioSample:
                return AudioSample(
                    waveform=[0.0],
                    sample_rate=32000,
                    duration_seconds=0.001,
                    file_path=file_path,
                )

        loader = _MockLoader()
        assert isinstance(loader, AudioLoader)

    def test_non_conforming_class_fails_check(self) -> None:
        """load メソッドを持たないクラスは Protocol を満たさないこと。"""

        class _BadLoader:
            pass

        assert not isinstance(_BadLoader(), AudioLoader)


class TestSpectrogramConfig:
    """SpectrogramConfig dataclass のテスト。"""

    def test_default_values(self) -> None:
        """デフォルト値が正しいこと。"""
        cfg = SpectrogramConfig()
        assert cfg.sample_rate == 32000
        assert cfg.n_fft == 2048
        assert cfg.hop_length == 512
        assert cfg.n_mels == 128
        assert cfg.segment_seconds == 5.0

    def test_segment_samples_computed(self) -> None:
        """segment_samples が sample_rate * segment_seconds であること。"""
        cfg = SpectrogramConfig(sample_rate=32000, segment_seconds=5.0)
        assert cfg.segment_samples == 160000

    def test_custom_values(self) -> None:
        """カスタム値で初期化できること。"""
        cfg = SpectrogramConfig(
            sample_rate=16000, n_fft=1024, hop_length=256, n_mels=64, segment_seconds=2.0
        )
        assert cfg.sample_rate == 16000
        assert cfg.segment_samples == 32000

    def test_is_frozen(self) -> None:
        """frozen dataclass であり属性変更不可であること。"""
        cfg = SpectrogramConfig()
        try:
            cfg.sample_rate = 1  # type: ignore[misc]
            raised = False
        except Exception:
            raised = True
        assert raised


class TestSpectrogramTransformerProtocol:
    """SpectrogramTransformer Protocol のテスト。"""

    def test_protocol_is_runtime_checkable(self) -> None:
        """SpectrogramTransformer が runtime_checkable であること。"""

        class _MockTransformer:
            def transform_file(self, file_path: str) -> list[list[float]]:
                return [[0.0]]

        assert isinstance(_MockTransformer(), SpectrogramTransformer)

    def test_non_conforming_class_fails_check(self) -> None:
        """transform_file メソッドを持たないクラスは Protocol を満たさないこと。"""

        class _BadTransformer:
            pass

        assert not isinstance(_BadTransformer(), SpectrogramTransformer)
