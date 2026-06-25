"""
notebook_loader — push_notebook usecase の competition 固有設定ロードのテスト。

なぜこのテストが必要か:
  - `conf/usecase/push_notebook.yaml` は routing-only 化したため、
    `cfg.recipe` から `conf/competition/{name}/notebook/{recipe}.yaml` を
    読み込んでマージする処理が必要になった。
  - 既存の `load_pipeline_recipe_cfg` / `load_trainer_cfgs` と同じパターンで実装するため、
    同じ振る舞い（recipe 未指定時は素通し、存在しない recipe は ValueError）をテストで固定する。
  - pipeline 経由（step で `notebook:` を直接 dict 指定）の場合は recipe が無いため、
    cfg をそのまま返すことを保証し、既存の pipeline 実行を壊さないようにする。
"""

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from src.usecase.notebook.notebook_loader import load_notebook_recipe_cfg


@pytest.fixture
def conf_dir(tmp_path: Path) -> Path:
    """competition/titanic/notebook/ に yaml ファイルを持つ仮の conf ディレクトリ。"""
    notebook_dir = tmp_path / "competition" / "titanic" / "notebook"
    notebook_dir.mkdir(parents=True)
    (notebook_dir / "all_after_download.yaml").write_text(
        "notebook:\n"
        "  competition: titanic\n"
        "  kernel_slug: titanic-pipeline\n"
        "  src_dataset: mlops-pipeline-src\n"
        "  recipe: all_after_download\n"
        "  enable_gpu: false\n"
        "  enable_internet: false\n"
        "  extra_datasets: []\n"
    )
    return tmp_path


@pytest.fixture
def base_cfg() -> DictConfig:
    return OmegaConf.create(
        {"competition": {"name": "titanic"}, "usecase": "push_notebook", "output_dir": "notebooks"}
    )


class TestLoadNotebookRecipeCfg:
    def test_returns_cfg_unchanged_when_recipe_is_none(
        self, conf_dir: Path, base_cfg: DictConfig
    ) -> None:
        """recipe 未指定（pipeline 経由で notebook が既に dict 指定済み）のときは素通しする。"""
        cfg = DictConfig(
            OmegaConf.merge(base_cfg, OmegaConf.create({"notebook": {"competition": "titanic"}}))
        )
        result = load_notebook_recipe_cfg(cfg, conf_dir)
        assert result.notebook.competition == "titanic"

    def test_merges_competition_specific_notebook_yaml_when_recipe_specified(
        self, conf_dir: Path, base_cfg: DictConfig
    ) -> None:
        """recipe 指定時は conf/competition/{name}/notebook/{recipe}.yaml をマージする。"""
        cfg = DictConfig(
            OmegaConf.merge(base_cfg, OmegaConf.create({"recipe": "all_after_download"}))
        )
        result = load_notebook_recipe_cfg(cfg, conf_dir)
        assert result.notebook.competition == "titanic"
        assert result.notebook.kernel_slug == "titanic-pipeline"

    def test_raises_when_specified_recipe_not_found(
        self, conf_dir: Path, base_cfg: DictConfig
    ) -> None:
        """存在しない recipe を指定したときは ValueError。"""
        cfg = DictConfig(OmegaConf.merge(base_cfg, OmegaConf.create({"recipe": "nonexistent"})))
        with pytest.raises(ValueError, match="nonexistent"):
            load_notebook_recipe_cfg(cfg, conf_dir)

    def test_base_cfg_keys_are_merged(self, conf_dir: Path, base_cfg: DictConfig) -> None:
        """base_cfg のキー（output_dir など）がマージ結果に含まれること。"""
        cfg = DictConfig(
            OmegaConf.merge(base_cfg, OmegaConf.create({"recipe": "all_after_download"}))
        )
        result = load_notebook_recipe_cfg(cfg, conf_dir)
        assert result.output_dir == "notebooks"
