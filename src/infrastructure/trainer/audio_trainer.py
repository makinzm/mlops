"""
AudioTrainer — Trainer Protocol の音声分類モデル実装。

メルスペクトログラム + CNN backbone（既定: EfficientNet-B0）による
multi-label / multi-class 分類。BCEWithLogitsLoss + macro-averaged ROC-AUC で
fold ごとに学習する。

入力は前処理 usecase の出力（manifest.json, cv_splits.json）を想定する。
manifest の各要素は以下の形式:
    {"file_path": str, "label": list[float]}
file_path が .pt の場合は torch.load で読み込み、1D（波形）なら
transformer で変換、2D（事前計算済みメルスペクトログラム）ならそのまま使う。
.pt 以外（.wav 等）の場合は transformer で on-demand 変換する。

backbone・num_classes は cfg 経由で指定する（ハードコードしない）。

時間計算量: O(F * E * N * C) — F: fold数, E: epoch数, N: サンプル数, C: モデル計算量
空間計算量: O(P + B * n_mels * n_frames) — P: パラメータ数, B: バッチサイズ
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.domain.data.audio import SpectrogramConfig
from src.domain.model.seed import SeedFixer
from src.domain.model.trainer import FoldResult, TrainResult
from src.infrastructure.audio.mel_spectrogram import MelSpectrogramTransformer
from src.infrastructure.trainer.torch_utils.audio_augmentation import mixup, spec_augment
from src.infrastructure.trainer.torch_utils.seed import TorchSeedFixer

logger = logging.getLogger(__name__)


class _ManifestDataset(Dataset):
    """manifest ベースの音声 Dataset。

    事前計算済みスペクトログラム(.pt) を torch.load で読み込む。
    .pt が波形テンソル（1D）の場合は transformer で変換する。
    音声ファイル（.wav 等）の場合は transformer で on-demand 変換する。

    時間計算量: __getitem__ は O(1)（事前計算済み .pt）または O(n * n_fft)（音声ファイル）
    空間計算量: O(n_mels * n_frames) — サンプルあたり
    """

    def __init__(
        self,
        manifest: list[dict[str, Any]],
        indices: list[int],
        transformer: MelSpectrogramTransformer | None = None,
    ) -> None:
        self._items = [manifest[i] for i in indices]
        self._transformer = transformer

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """スペクトログラムとラベルを返す。

        時間計算量: O(1)（事前計算済み .pt）または O(n * n_fft)（音声ファイル）
        空間計算量: O(n_mels * n_frames)
        """
        item = self._items[index]
        file_path = item["file_path"]

        if file_path.endswith(".pt"):
            mel = torch.load(file_path, weights_only=True)
            if mel.dim() == 1 and self._transformer:
                mel = self._transformer.transform(mel)
        else:
            if self._transformer is None:
                msg = "transformer required for audio files"
                raise ValueError(msg)
            mel = self._transformer.transform_file(file_path)

        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        label = torch.tensor(item["label"], dtype=torch.float32)
        return mel, label


def _build_audio_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """音声分類モデルを構築する。

    backbone の入力チャネルを 1ch（メルスペクトログラム）に変更し、
    分類ヘッドを num_classes に設定する。

    Args:
        backbone: backbone 名。現在 "efficientnet_b0" のみサポート。
        num_classes: 分類クラス数。
        pretrained: ImageNet pretrained 重みを使うか。

    Returns:
        nn.Module: 構築したモデル。

    時間計算量: O(1)
    空間計算量: O(P) — P: モデルパラメータ数
    """
    if backbone == "efficientnet_b0":
        from torchvision.models import efficientnet_b0

        weights = "IMAGENET1K_V1" if pretrained else None
        model = efficientnet_b0(weights=weights)
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    msg = f"Unknown backbone: {backbone!r}"
    raise ValueError(msg)


def _compute_macro_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """macro-averaged ROC-AUC を計算する。

    全陰性（col_sum == 0）および全陽性（col_sum == n_samples）のクラスは
    sklearn が ValueError を送出するためスキップする。

    時間計算量: O(C * N) — C: スコア対象クラス数, N: サンプル数
    空間計算量: O(C * N)
    """
    n_samples = y_true.shape[0]
    col_sums = y_true.sum(axis=0)
    scored_cols = np.where((col_sums > 0) & (col_sums < n_samples))[0]
    if len(scored_cols) == 0:
        return 0.0
    return float(
        roc_auc_score(
            y_true[:, scored_cols],
            y_pred[:, scored_cols],
            average="macro",
        )
    )


class AudioTrainer:
    """音声分類モデルの k-fold クロスバリデーション学習器。

    Trainer Protocol を満たす。SeedFixer はコンストラクタで DI し、
    デフォルトは TorchSeedFixer を使う（VisionTrainer と同じパターン）。
    """

    def __init__(self, seed_fixer: SeedFixer | None = None) -> None:
        self._seed_fixer: SeedFixer = seed_fixer if seed_fixer is not None else TorchSeedFixer()

    def fit_folds(
        self,
        preprocess_output_dir: Path,
        output_dir: Path,
        cfg: dict[str, Any],
    ) -> TrainResult:
        """fold ごとに学習し TrainResult を返す。

        時間計算量: O(F * E * N * C)
        空間計算量: O(P + B * n_mels * n_frames)
        """
        timestamp = cfg.get("_timestamp", datetime.now().strftime("%Y%m%dT%H%M%S"))
        commit_hash: str = cfg.get("_commit_hash", "unknown")
        job_id: str = cfg.get("job_id", "audio")
        seed: int = int(cfg.get("seed", 42))

        spec_cfg = cfg.get("spectrogram", {})
        config = SpectrogramConfig(
            sample_rate=spec_cfg.get("sample_rate", 32000),
            n_fft=spec_cfg.get("n_fft", 2048),
            hop_length=spec_cfg.get("hop_length", 512),
            n_mels=spec_cfg.get("n_mels", 128),
            segment_seconds=spec_cfg.get("segment_seconds", 5.0),
        )
        transformer = MelSpectrogramTransformer(config)

        model_cfg = cfg.get("model", {})
        backbone: str = model_cfg.get("backbone", "efficientnet_b0")
        pretrained: bool = model_cfg.get("pretrained", True)
        if "num_classes" not in model_cfg:
            msg = "cfg['model']['num_classes'] は必須です。分類クラス数を指定してください。"
            raise ValueError(msg)
        num_classes: int = int(model_cfg["num_classes"])

        train_cfg = cfg.get("training", {})
        num_epochs: int = int(train_cfg.get("epochs", 20))
        batch_size: int = int(train_cfg.get("batch_size", 32))
        lr: float = float(train_cfg.get("lr", 1e-3))
        weight_decay: float = float(train_cfg.get("weight_decay", 1e-4))

        aug_cfg = cfg.get("augmentation", {})
        use_spec_augment: bool = bool(aug_cfg.get("spec_augment", False))
        use_mixup: bool = bool(aug_cfg.get("mixup", False))
        mixup_alpha: float = float(aug_cfg.get("mixup_alpha", 0.4))
        freq_mask_param: int = int(aug_cfg.get("freq_mask_param", 20))
        time_mask_param: int = int(aug_cfg.get("time_mask_param", 40))

        with open(preprocess_output_dir / "manifest.json") as f:
            manifest = json.load(f)
        with open(preprocess_output_dir / "cv_splits.json") as f:
            splits = json.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fold_results: list[FoldResult] = []

        for fold_idx, split in enumerate(splits):
            fold_out = output_dir / f"fold_{fold_idx}"
            fold_out.mkdir(parents=True, exist_ok=True)

            self._seed_fixer.fix(seed)
            logger.info("Fold %d/%d", fold_idx, len(splits))

            train_ds = _ManifestDataset(manifest, split["train"], transformer)
            val_ds = _ManifestDataset(manifest, split["val"], transformer)

            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            )
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            model = _build_audio_model(backbone, num_classes, pretrained).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            best_score = 0.0
            best_state: OrderedDict[str, torch.Tensor] = OrderedDict()
            final_train_loss = 0.0

            for epoch in range(num_epochs):
                model.train()
                train_loss_sum = 0.0
                train_count = 0
                for mel_batch, label_batch in train_loader:
                    mel_batch, label_batch = mel_batch.to(device), label_batch.to(device)
                    if use_spec_augment:
                        mel_batch = spec_augment(
                            mel_batch,
                            freq_mask_param=freq_mask_param,
                            time_mask_param=time_mask_param,
                        )
                    if use_mixup:
                        mel_batch, label_batch = mixup(mel_batch, label_batch, alpha=mixup_alpha)
                    optimizer.zero_grad()
                    out = model(mel_batch)
                    loss = criterion(out, label_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss_sum += loss.item() * mel_batch.size(0)
                    train_count += mel_batch.size(0)

                final_train_loss = train_loss_sum / max(train_count, 1)

                model.eval()
                all_preds: list[np.ndarray] = []
                all_labels: list[np.ndarray] = []
                with torch.no_grad():
                    for mel_batch, label_batch in val_loader:
                        mel_batch = mel_batch.to(device)
                        out = model(mel_batch)
                        probs = torch.sigmoid(out).cpu().numpy()
                        all_preds.append(probs)
                        all_labels.append(label_batch.numpy())

                y_pred = np.concatenate(all_preds, axis=0)
                y_true = np.concatenate(all_labels, axis=0)

                val_score = _compute_macro_roc_auc(y_true, y_pred)

                if val_score >= best_score:
                    best_score = val_score
                    best_state = OrderedDict({k: v.clone() for k, v in model.state_dict().items()})

                logger.info(
                    "  Epoch %d/%d: loss=%.4f, roc_auc=%.4f",
                    epoch + 1,
                    num_epochs,
                    final_train_loss,
                    val_score,
                )

            model_path = fold_out / "model.pt"
            torch.save(
                {
                    "model_state_dict": best_state,
                    "backbone": backbone,
                    "num_classes": num_classes,
                    "spectrogram_config": {
                        "sample_rate": config.sample_rate,
                        "n_fft": config.n_fft,
                        "hop_length": config.hop_length,
                        "n_mels": config.n_mels,
                        "segment_seconds": config.segment_seconds,
                    },
                },
                model_path,
            )

            oof_path = fold_out / "oof_predictions.npy"
            np.save(oof_path, y_pred)

            ea_path = fold_out / "error_analysis.npy"
            np.save(ea_path, np.stack([y_true, y_pred], axis=0))

            fold_results.append(
                FoldResult(
                    fold_idx=fold_idx,
                    train_score=final_train_loss,
                    valid_score=best_score,
                    metric="roc_auc",
                    model_path=model_path,
                    oof_path=oof_path,
                    error_analysis_path=ea_path,
                    feature_importance_path=None,
                    n_train=len(split["train"]),
                    n_valid=len(split["val"]),
                )
            )

        scores = [f.valid_score for f in fold_results]
        return TrainResult(
            job_id=job_id,
            timestamp=timestamp,
            commit_hash=commit_hash,
            trainer_type="audio",
            output_dir=output_dir,
            fold_results=fold_results,
            cv_mean_score=float(np.mean(scores)),
            cv_std_score=float(np.std(scores)),
            metric="roc_auc",
            seed=seed,
        )
