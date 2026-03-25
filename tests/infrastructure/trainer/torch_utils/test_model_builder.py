"""
torch_utils/model_builder のテスト。

なぜこのテストが必要か:
  - build_vision_model() が BackboneConfig から nn.Module を構築すること。
  - save_checkpoint() / load_checkpoint() でモデルの保存・復元が正しく動くこと。
  - 全ての Vision 関連コードで checkpoint 形式が統一されること。

時間計算量: O(P) — P: パラメータ数
空間計算量: O(P)
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.domain.model.backbone import BackboneConfig
from src.infrastructure.trainer.torch_utils.model_builder import (
    build_vision_model,
    load_checkpoint,
    save_checkpoint,
)


class TestBuildVisionModel:
    def test_returns_module_with_correct_output(self) -> None:
        """build_vision_model が正しい出力次元の nn.Module を返すこと。"""
        config = BackboneConfig(backbone_name="simple_cnn", num_classes=2, pretrained=False)
        model = build_vision_model(config)
        assert isinstance(model, nn.Module)
        dummy = torch.randn(1, 3, 32, 32)
        output = model(dummy)
        assert output.shape == (1, 2)


class TestCheckpointSaveLoad:
    def test_roundtrip(self, tmp_path: Path) -> None:
        """save → load で同じ予測結果が復元されること。"""
        config = BackboneConfig(
            backbone_name="simple_cnn", num_classes=2, pretrained=False, image_size=32
        )
        model = build_vision_model(config)
        model.eval()
        dummy = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            original_out = model(dummy)

        path = tmp_path / "model.pt"
        save_checkpoint(model, config, path)

        loaded_model = load_checkpoint(path, device=torch.device("cpu"))
        loaded_model.eval()
        with torch.no_grad():
            loaded_out = loaded_model(dummy)

        assert torch.allclose(original_out, loaded_out)
