"""
AudioTrainer のテスト。

なぜこのテストが必要か:
- 音声分類モデルが Trainer Protocol を満たすことを検証する。
- メルスペクトログラム入力（1ch）で forward pass が動作することを確認する。
- BCEWithLogitsLoss + ROC-AUC による multi-label 学習が正しく動くことを保証する。
- fit_folds が TrainResult を返すことを確認する。
- cfg 経由で num_classes / backbone を指定できること（BirdCLEF 固有のハードコード除去）を保証する。
- SeedFixer Protocol 経由で seed 固定されること（caution.md: seed 固定関数の直接呼び出し禁止）を保証する。
"""

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from src.domain.model.seed import SeedFixer
from src.domain.model.trainer import Trainer, TrainResult
from src.infrastructure.trainer.audio_trainer import AudioTrainer


class _SpySeedFixer:
    """SeedFixer Protocol を満たすテスト用スパイ。呼び出し回数・引数を記録する。"""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def fix(self, seed: int) -> None:
        self.calls.append(seed)


@pytest.fixture()
def audio_data_dir(tmp_path: Path) -> Path:
    """AudioTrainer 用のテストデータを作成する。

    前処理 usecase の出力（manifest.json, cv_splits.json）と同じ構造を模倣する。
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    num_classes = 3
    manifest = []
    for i in range(12):
        class_idx = i % num_classes
        label = [0.0] * num_classes
        label[class_idx] = 1.0
        audio_path = audio_dir / f"sample_{i}.pt"
        # 5秒分のダミー波形を保存（AudioTrainer 内で transform される）
        torch.save(torch.randn(160000), audio_path)
        manifest.append({"file_path": str(audio_path), "label": label})

    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    splits = [
        {"train": list(range(6)), "val": list(range(6, 12))},
        {"train": list(range(6, 12)), "val": list(range(6))},
    ]
    splits_path = tmp_path / "cv_splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f)

    return tmp_path


@pytest.fixture()
def train_cfg() -> dict[str, Any]:
    """学習設定を返す。num_classes はテストデータ（3クラス）に合わせる。"""
    return {
        "job_id": "test_audio",
        "trainer_type": "audio",
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
            "pretrained": False,  # テストでは pretrained=False で高速化
            "num_classes": 3,
        },
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
    }


class TestAudioTrainer:
    """AudioTrainer のテスト。"""

    def test_implements_trainer_protocol(self) -> None:
        """Trainer Protocol を満たすこと。"""
        trainer = AudioTrainer()
        assert isinstance(trainer, Trainer)

    def test_fit_folds_returns_train_result(
        self, audio_data_dir: Path, train_cfg: dict[str, Any], tmp_path: Path
    ) -> None:
        """fit_folds が TrainResult を返すこと。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        trainer = AudioTrainer()
        result = trainer.fit_folds(
            preprocess_output_dir=audio_data_dir,
            output_dir=output_dir,
            cfg=train_cfg,
        )

        assert isinstance(result, TrainResult)
        assert result.trainer_type == "audio"
        assert result.seed == 42
        assert len(result.fold_results) == 2

    def test_fold_results_have_scores(
        self, audio_data_dir: Path, train_cfg: dict[str, Any], tmp_path: Path
    ) -> None:
        """各 fold の結果にスコアが含まれ、モデルが保存されること。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        trainer = AudioTrainer()
        result = trainer.fit_folds(
            preprocess_output_dir=audio_data_dir,
            output_dir=output_dir,
            cfg=train_cfg,
        )

        for fold_result in result.fold_results:
            assert fold_result.valid_score >= 0.0
            assert fold_result.valid_score <= 1.0
            assert fold_result.model_path.exists()

    def test_model_checkpoint_saved(
        self, audio_data_dir: Path, train_cfg: dict[str, Any], tmp_path: Path
    ) -> None:
        """モデルチェックポイントが保存され、ロード可能であること。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        trainer = AudioTrainer()
        result = trainer.fit_folds(
            preprocess_output_dir=audio_data_dir,
            output_dir=output_dir,
            cfg=train_cfg,
        )

        for fold_result in result.fold_results:
            assert fold_result.model_path.exists()
            checkpoint = torch.load(fold_result.model_path, weights_only=True)
            assert "model_state_dict" in checkpoint

    def test_uses_injected_seed_fixer(
        self, audio_data_dir: Path, train_cfg: dict[str, Any], tmp_path: Path
    ) -> None:
        """SeedFixer Protocol 経由で seed が固定されること（fold 数だけ呼ばれる）。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        spy = _SpySeedFixer()
        trainer = AudioTrainer(seed_fixer=spy)
        trainer.fit_folds(
            preprocess_output_dir=audio_data_dir,
            output_dir=output_dir,
            cfg=train_cfg,
        )

        assert spy.calls == [42, 42]

    def test_num_classes_from_cfg(
        self, audio_data_dir: Path, train_cfg: dict[str, Any], tmp_path: Path
    ) -> None:
        """num_classes がハードコードでなく cfg から取られること。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        train_cfg["model"]["num_classes"] = 3

        trainer = AudioTrainer()
        result = trainer.fit_folds(
            preprocess_output_dir=audio_data_dir,
            output_dir=output_dir,
            cfg=train_cfg,
        )

        checkpoint = torch.load(result.fold_results[0].model_path, weights_only=True)
        assert checkpoint["num_classes"] == 3

    def test_missing_num_classes_raises(
        self, audio_data_dir: Path, train_cfg: dict[str, Any], tmp_path: Path
    ) -> None:
        """num_classes が cfg に無い場合は明確なエラーになること。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        del train_cfg["model"]["num_classes"]

        trainer = AudioTrainer()
        with pytest.raises(ValueError, match="num_classes"):
            trainer.fit_folds(
                preprocess_output_dir=audio_data_dir,
                output_dir=output_dir,
                cfg=train_cfg,
            )
