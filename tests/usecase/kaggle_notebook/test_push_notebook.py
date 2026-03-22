"""
PushNotebookUseCase のテスト。

なぜこのテストが必要か:
  - PushNotebookUseCase は Notebook 生成 → kernel-metadata.json 生成 → Kaggle プッシュを担う。
  - Notebook のセル構成（3セル）・各セルの内容・kernel-metadata.json の必須フィールドが
    正しく生成されることをテストで保証しないと、Kaggle 上で Notebook が動かなくなる。
  - Kaggle API 呼び出しは MagicMock で差し替え、CI で実際の通信が発生しないようにする。
  - per-directory .gitignore と README.md の生成も確認する（DoD 要件）。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from src.usecase.kaggle_notebook.push_notebook import PushNotebookUseCase


def _make_cfg(tmp_path: Path) -> object:
    """PushNotebookUseCase 用の DictConfig を生成する。"""
    return OmegaConf.create(
        {
            "usecase": "push_notebook",
            "notebook": {
                "competition": "titanic",
                "kernel_slug": "titanic-pipeline",
                "src_dataset": "mlops-pipeline-src",
                "enable_gpu": False,
                "enable_internet": True,
            },
            "output_dir": str(tmp_path / "push_notebook"),
            "kaggle_username": "testuser",
        }
    )


class TestNotebookGeneration:
    """Notebook ファイルの生成を検証する。"""

    def test_execute_creates_notebook_ipynb(self, tmp_path: Path) -> None:
        """execute() が notebook.ipynb を生成すること。"""
        cfg = _make_cfg(tmp_path)
        mock_api = MagicMock()
        usecase = PushNotebookUseCase(cfg=cfg, kaggle_api=mock_api)  # type: ignore[arg-type]
        result = usecase.execute()
        assert result.notebook_path.exists(), "notebook.ipynb が生成されていない"
        assert result.notebook_path.name == "notebook.ipynb"

    def test_generated_notebook_has_three_cells(self, tmp_path: Path) -> None:
        """生成された Notebook が3セルで構成されること。"""
        cfg = _make_cfg(tmp_path)
        mock_api = MagicMock()
        usecase = PushNotebookUseCase(cfg=cfg, kaggle_api=mock_api)  # type: ignore[arg-type]
        result = usecase.execute()
        notebook = json.loads(result.notebook_path.read_text())
        cells = notebook["cells"]
        assert len(cells) == 3, f"セル数が期待と異なる: {len(cells)}"

    def test_generated_notebook_cell_types_are_code(self, tmp_path: Path) -> None:
        """全セルが code タイプであること。"""
        cfg = _make_cfg(tmp_path)
        mock_api = MagicMock()
        usecase = PushNotebookUseCase(cfg=cfg, kaggle_api=mock_api)  # type: ignore[arg-type]
        result = usecase.execute()
        notebook = json.loads(result.notebook_path.read_text())
        for i, cell in enumerate(notebook["cells"]):
            assert cell["cell_type"] == "code", (
                f"セル{i + 1} が code タイプでない: {cell['cell_type']}"
            )

    def test_generated_notebook_cell1_contains_pip_install(self, tmp_path: Path) -> None:
        """セル1に pip install が含まれること（環境セットアップセル）。"""
        cfg = _make_cfg(tmp_path)
        mock_api = MagicMock()
        usecase = PushNotebookUseCase(cfg=cfg, kaggle_api=mock_api)  # type: ignore[arg-type]
        result = usecase.execute()
        notebook = json.loads(result.notebook_path.read_text())
        cell1_source = "".join(notebook["cells"][0]["source"])
        assert "pip install" in cell1_source, "セル1に pip install が含まれない"

    def test_generated_notebook_cell1_smart_install_filters_indented_comments(
        self, tmp_path: Path
    ) -> None:
        """セル1の _smart_install が先頭スペース付きコメント行（`    # via aiohttp`）を
        除外すること。`uv export` は `    # via pkg` のようにインデントされたコメントを
        生成するため、strip() 後の startswith('#') で判定しないと pip に渡されてしまう。"""
        cfg = _make_cfg(tmp_path)
        mock_api = MagicMock()
        usecase = PushNotebookUseCase(cfg=cfg, kaggle_api=mock_api)  # type: ignore[arg-type]
        result = usecase.execute()
        notebook = json.loads(result.notebook_path.read_text())
        cell1_source = "".join(notebook["cells"][0]["source"])
        # strip().startswith が使われていることを確認（元行に先頭スペースがあっても除外できる）
        assert "l.strip().startswith('#')" in cell1_source, (
            "_smart_install が l.strip().startswith('#') を使っていない — "
            "uv export の `    # via aiohttp` がpipに渡されてしまう"
        )

    def test_generated_notebook_cell2_contains_competition_slug(self, tmp_path: Path) -> None:
        """セル2に competition slug（titanic）が含まれること（設定上書きセル）。"""
        cfg = _make_cfg(tmp_path)
        mock_api = MagicMock()
        usecase = PushNotebookUseCase(cfg=cfg, kaggle_api=mock_api)  # type: ignore[arg-type]
        result = usecase.execute()
        notebook = json.loads(result.notebook_path.read_text())
        cell2_source = "".join(notebook["cells"][1]["source"])
        assert "titanic" in cell2_source, "セル2に competition slug が含まれない"

    def test_generated_notebook_cell3_contains_notebook_pipeline_runner(
        self, tmp_path: Path
    ) -> None:
        """セル3に NotebookPipelineRunner の呼び出しが含まれること（実行セル）。"""
        cfg = _make_cfg(tmp_path)
        mock_api = MagicMock()
        usecase = PushNotebookUseCase(cfg=cfg, kaggle_api=mock_api)  # type: ignore[arg-type]
        result = usecase.execute()
        notebook = json.loads(result.notebook_path.read_text())
        cell3_source = "".join(notebook["cells"][2]["source"])
        assert "NotebookPipelineRunner" in cell3_source, (
            "セル3に NotebookPipelineRunner が含まれない"
        )


class TestKernelMetadataGeneration:
    """kernel-metadata.json の生成を検証する。"""

    def test_execute_creates_kernel_metadata_json(self, tmp_path: Path) -> None:
        """execute() が kernel-metadata.json を生成すること。"""
        cfg = _make_cfg(tmp_path)
        mock_api = MagicMock()
        usecase = PushNotebookUseCase(cfg=cfg, kaggle_api=mock_api)  # type: ignore[arg-type]
        result = usecase.execute()
        assert result.metadata_path.exists(), "kernel-metadata.json が生成されていない"
        assert result.metadata_path.name == "kernel-metadata.json"

    def test_kernel_metadata_has_required_fields(self, tmp_path: Path) -> None:
        """kernel-metadata.json に必須フィールドが含まれること。"""
        cfg = _make_cfg(tmp_path)
        mock_api = MagicMock()
        usecase = PushNotebookUseCase(cfg=cfg, kaggle_api=mock_api)  # type: ignore[arg-type]
        result = usecase.execute()
        metadata = json.loads(result.metadata_path.read_text())
        for field in ("id", "language", "kernel_type", "is_private", "code_file"):
            assert field in metadata, f"kernel-metadata.json に '{field}' が含まれない"

    def test_kernel_metadata_competition_sources(self, tmp_path: Path) -> None:
        """kernel-metadata.json の competition_sources に competition slug が設定されること。"""
        cfg = _make_cfg(tmp_path)
        mock_api = MagicMock()
        usecase = PushNotebookUseCase(cfg=cfg, kaggle_api=mock_api)  # type: ignore[arg-type]
        result = usecase.execute()
        metadata = json.loads(result.metadata_path.read_text())
        assert metadata.get("competition_sources") == ["titanic"], (
            f"competition_sources が期待と異なる: {metadata.get('competition_sources')}"
        )

    def test_kernel_metadata_dataset_sources(self, tmp_path: Path) -> None:
        """kernel-metadata.json の dataset_sources に src_dataset が設定されること。"""
        cfg = _make_cfg(tmp_path)
        mock_api = MagicMock()
        usecase = PushNotebookUseCase(cfg=cfg, kaggle_api=mock_api)  # type: ignore[arg-type]
        result = usecase.execute()
        metadata = json.loads(result.metadata_path.read_text())
        assert "testuser/mlops-pipeline-src" in metadata.get("dataset_sources", []), (
            f"dataset_sources に src_dataset が含まれない: {metadata.get('dataset_sources')}"
        )


class TestKaggleApiInteraction:
    """Kaggle API との連携を検証する。"""

    def test_execute_calls_kernels_push(self, tmp_path: Path) -> None:
        """execute() が Kaggle API の kernels_push() を1回呼ぶこと。"""
        cfg = _make_cfg(tmp_path)
        mock_api = MagicMock()
        usecase = PushNotebookUseCase(cfg=cfg, kaggle_api=mock_api)  # type: ignore[arg-type]
        usecase.execute()
        mock_api.kernels_push.assert_called_once()

    def test_execute_raises_runtime_error_on_auth_failure(self, tmp_path: Path) -> None:
        """Kaggle API が SystemExit を上げたとき RuntimeError に変換されること。"""
        cfg = _make_cfg(tmp_path)
        mock_api = MagicMock()
        mock_api.kernels_push.side_effect = SystemExit(1)
        usecase = PushNotebookUseCase(cfg=cfg, kaggle_api=mock_api)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="Kaggle"):
            usecase.execute()


class TestOutputFiles:
    """出力ファイルの生成を検証する（DoD 要件）。"""

    def test_execute_creates_gitignore(self, tmp_path: Path) -> None:
        """execute() が per-directory .gitignore を生成すること。"""
        cfg = _make_cfg(tmp_path)
        mock_api = MagicMock()
        usecase = PushNotebookUseCase(cfg=cfg, kaggle_api=mock_api)  # type: ignore[arg-type]
        result = usecase.execute()
        gitignore = result.notebook_path.parent / ".gitignore"
        assert gitignore.exists(), ".gitignore が生成されていない"

    def test_execute_creates_readme(self, tmp_path: Path) -> None:
        """execute() が README.md を生成すること。"""
        cfg = _make_cfg(tmp_path)
        mock_api = MagicMock()
        usecase = PushNotebookUseCase(cfg=cfg, kaggle_api=mock_api)  # type: ignore[arg-type]
        result = usecase.execute()
        readme = result.notebook_path.parent / "README.md"
        assert readme.exists(), "README.md が生成されていない"


class TestNoPipNotebook:
    """enable_internet=False のとき pip install がスキップされることを検証する。

    なぜこのテストが必要か:
      - インターネットオフのコンペでは pip install が実行できないため、
        Cell 1 に _smart_install 呼び出しが含まれてはならない。
      - enable_internet フラグをテンプレートに渡すことで対応し、
        自動テストで退行を防ぐ。
    """

    def test_cell1_has_smart_install_when_internet_enabled(self, tmp_path: Path) -> None:
        """enable_internet=True（デフォルト）のとき Cell 1 に _smart_install が含まれること。"""
        cfg = _make_cfg(tmp_path)
        mock_api = MagicMock()
        usecase = PushNotebookUseCase(cfg=cfg, kaggle_api=mock_api)  # type: ignore[arg-type]
        result = usecase.execute()
        notebook = json.loads(result.notebook_path.read_text())
        cell1_source = "".join(notebook["cells"][0]["source"])
        assert "_smart_install(" in cell1_source, (
            "enable_internet=True なのに Cell 1 に _smart_install 呼び出しがない"
        )

    def test_cell1_skips_smart_install_when_internet_disabled(self, tmp_path: Path) -> None:
        """enable_internet=False のとき Cell 1 に _smart_install 呼び出しがないこと。"""
        from omegaconf import OmegaConf

        raw = {
            "usecase": "push_notebook",
            "notebook": {
                "competition": "titanic",
                "kernel_slug": "titanic-pipeline",
                "src_dataset": "mlops-pipeline-src",
                "enable_gpu": False,
                "enable_internet": False,  # ← オフライン
            },
            "output_dir": str(tmp_path / "push_notebook"),
            "kaggle_username": "testuser",
        }
        cfg = OmegaConf.create(raw)
        mock_api = MagicMock()
        usecase = PushNotebookUseCase(cfg=cfg, kaggle_api=mock_api)  # type: ignore[arg-type]
        result = usecase.execute()
        notebook = json.loads(result.notebook_path.read_text())
        cell1_source = "".join(notebook["cells"][0]["source"])
        assert "_smart_install(" not in cell1_source, (
            "enable_internet=False なのに Cell 1 に _smart_install 呼び出しが含まれている"
        )
