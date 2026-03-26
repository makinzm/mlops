"""
presentation/registry モジュールのテスト。

なぜこのテストが必要か:
  - main.py の巨大な if/elif チェーン (13 分岐) を registry パターンに置き換える。
  - dispatch() が正しい runner を呼び出すこと、
    未知の usecase で ValueError を上げることを検証する。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf


class TestDispatch:
    """dispatch() のテスト。"""

    def test_dispatches_known_usecase(self) -> None:
        """登録済み usecase に対して対応する runner が呼ばれること。"""
        from src.presentation.registry import dispatch

        cfg = OmegaConf.create({"usecase": "download_dataset"})
        mock_logger = MagicMock()
        mock_runner = MagicMock()

        dispatch("download_dataset", cfg, mock_logger, overrides={"download_dataset": mock_runner})
        mock_runner.assert_called_once_with(cfg, mock_logger)

    def test_raises_on_unknown_usecase(self) -> None:
        """未知の usecase で ValueError が発生すること。"""
        from src.presentation.registry import dispatch

        cfg = OmegaConf.create({"usecase": "unknown"})
        mock_logger = MagicMock()

        with pytest.raises(ValueError, match="Unknown usecase"):
            dispatch("unknown", cfg, mock_logger)

    def test_all_usecases_registered(self) -> None:
        """全ての usecase が RUNNERS レジストリに登録されていること。"""
        from src.presentation.registry import RUNNERS

        expected = {
            "download_dataset",
            "automatically_eda",
            "preprocess",
            "train",
            "inference",
            "remote_train",
            "vertex_submit",
            "vertex_download",
            "pipeline",
            "push_notebook",
            "create_source_dataset",
            "update_source_dataset",
            "gradcam",
        }
        assert set(RUNNERS.keys()) == expected
