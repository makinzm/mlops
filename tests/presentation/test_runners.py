"""
run_train のトレーナー型ディスパッチのテスト。

なぜこのテストが必要か:
- run_train が trainer.type=audio のとき AudioTrainer にディスパッチすることを検証する。
- trainer.type が未登録のとき ValueError が発生することを確認する。
- audio レシピ yaml が正しい trainer.type 構造を持つことを保証する。

このテストがない場合、conf yaml に trainer_type: audio（フラットキー）と
書いてもエラーが出ず、runner は ValueError を投げるだけになる。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig, OmegaConf


def _make_trainer_cfg(tmp_path: Path, trainer_type: str) -> DictConfig:
    """run_train に渡す最小限の trainer DictConfig を生成する。"""
    return OmegaConf.create(
        {
            "trainer": {"type": trainer_type},
            "preprocess_output_dir": str(tmp_path / "preprocess"),
            "output_dir": str(tmp_path / "models"),
            "job_id": f"test_{trainer_type}",
            "competition": {"name": "audio_example"},
            "seed": 42,
            "metric": "roc_auc",
            "spectrogram": {
                "sample_rate": 32000,
                "n_fft": 2048,
                "hop_length": 512,
                "n_mels": 128,
                "segment_seconds": 5.0,
            },
            "model": {
                "backbone": "efficientnet_b0",
                "pretrained": False,
                "num_classes": 3,
            },
            "training": {"epochs": 1, "batch_size": 4, "lr": 1e-3, "weight_decay": 1e-4},
        }
    )


class TestRunTrainAudioDispatch:
    """run_train の audio ディスパッチテスト。"""

    def test_audio_trainer_is_dispatched(self, tmp_path: Path) -> None:
        """trainer.type=audio のとき AudioTrainer が使われ ValueError が発生しないこと。"""
        from src.presentation.runners import run_train

        cfg = OmegaConf.create({"competition": {"name": "audio_example"}, "usecase": "train"})
        trainer_cfg = _make_trainer_cfg(tmp_path, "audio")

        mock_result = MagicMock()
        mock_result.job_id = "test_audio"
        mock_result.metric = "roc_auc"
        mock_result.cv_mean_score = 0.7
        mock_result.cv_std_score = 0.05

        with (
            patch(
                "src.usecase.training.trainer_loader.load_trainer_cfgs",
                return_value=[trainer_cfg],
            ),
            patch("src.infrastructure.repository.git.GitRepositoryImpl"),
            patch("src.usecase.training.train.TrainUseCase") as mock_uc,
        ):
            mock_uc.return_value.execute.return_value = mock_result
            run_train(cfg)

        mock_uc.assert_called_once()
        _, kwargs = mock_uc.call_args
        from src.infrastructure.trainer.audio_trainer import AudioTrainer

        assert isinstance(kwargs["trainer"], AudioTrainer)

    def test_unknown_trainer_type_raises(self, tmp_path: Path) -> None:
        """trainer.type が未登録の場合は ValueError が発生すること。"""
        from src.presentation.runners import run_train

        cfg = OmegaConf.create({"competition": {"name": "audio_example"}, "usecase": "train"})
        trainer_cfg = _make_trainer_cfg(tmp_path, "unknown_type")

        with (
            patch(
                "src.usecase.training.trainer_loader.load_trainer_cfgs",
                return_value=[trainer_cfg],
            ),
            patch("src.infrastructure.repository.git.GitRepositoryImpl"),
        ):
            with pytest.raises(ValueError, match="trainer.type='unknown_type'"):
                run_train(cfg)

    def test_audio_recipe_yaml_uses_nested_trainer_type(self) -> None:
        """efficientnet_b0.yaml が trainer.type: audio（ネスト形式）を使っていること。

        trainer_type: audio（フラットキー）では run_train がキーを読めずに
        AttributeError / KeyError を起こす。このテストがあれば yaml の構造ミスを早期検知できる。
        """
        conf_dir = Path(__file__).parent.parent.parent / "conf"
        yaml_path = conf_dir / "competition" / "audio_example" / "training" / "efficientnet_b0.yaml"
        assert yaml_path.exists(), f"yaml が見つかりません: {yaml_path}"
        cfg = OmegaConf.load(yaml_path)
        assert hasattr(cfg, "trainer"), (
            "trainer キーが存在しません（trainer_type ではなく trainer.type を使うこと）"
        )
        assert cfg.trainer.type == "audio", (
            f"trainer.type が audio ではありません: {cfg.trainer.type!r}"
        )
