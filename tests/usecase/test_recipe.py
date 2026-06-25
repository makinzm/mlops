"""
src/usecase/_recipe.py の共通 recipe ロードユーティリティのテスト。

なぜこのテストが必要か:
  - trainer_loader / inference_loader / preprocessing.pipeline_loader /
    usecase.pipeline.pipeline_loader / notebook_loader が同種の
    「conf/competition/{name}/{subdir}/{recipe}.yaml を cfg にマージする」
    ロジックをコピペしていたため、共通実装 load_recipe_cfgs / load_single_recipe_cfg
    に集約する。各 loader の既存テストは無修正で green を維持するために、
    フォールバックキー・空ディレクトリ時のメッセージ・必須/省略可の挙動を
    ここで個別に固定する。
"""

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from src.usecase._recipe import load_recipe_cfgs, load_single_recipe_cfg


@pytest.fixture
def conf_dir(tmp_path: Path) -> Path:
    """competition/titanic/training/ に yaml ファイルを持つ仮の conf ディレクトリ。"""
    training_dir = tmp_path / "competition" / "titanic" / "training"
    training_dir.mkdir(parents=True)
    (training_dir / "lgbm.yaml").write_text("trainer:\n  type: lgbm\n")
    (training_dir / "nn.yaml").write_text("trainer:\n  type: nn\n")
    return tmp_path


@pytest.fixture
def base_cfg() -> DictConfig:
    return OmegaConf.create({"competition": {"name": "titanic"}, "seed": 42})


class TestLoadRecipeCfgs:
    def test_returns_all_yamls_when_recipe_not_specified(
        self, conf_dir: Path, base_cfg: DictConfig
    ) -> None:
        """recipe 未指定時は subdir 配下の全 yaml をロードすること。"""
        cfgs = load_recipe_cfgs(base_cfg, "training", conf_dir)
        types = sorted(c.trainer.type for c in cfgs)
        assert types == ["lgbm", "nn"]

    def test_returns_single_yaml_when_recipe_specified(
        self, conf_dir: Path, base_cfg: DictConfig
    ) -> None:
        """recipe 指定時は該当 yaml のみロードすること。"""
        cfg = DictConfig(OmegaConf.merge(base_cfg, OmegaConf.create({"recipe": "lgbm"})))
        cfgs = load_recipe_cfgs(cfg, "training", conf_dir)
        assert len(cfgs) == 1
        assert cfgs[0].trainer.type == "lgbm"

    def test_uses_fallback_key_when_recipe_not_specified(
        self, conf_dir: Path, base_cfg: DictConfig
    ) -> None:
        """recipe 未指定だが fallback_key が指定されたとき、それを優先すること。"""
        cfg = DictConfig(OmegaConf.merge(base_cfg, OmegaConf.create({"trainer_name": "nn"})))
        cfgs = load_recipe_cfgs(cfg, "training", conf_dir, fallback_key="trainer_name")
        assert len(cfgs) == 1
        assert cfgs[0].trainer.type == "nn"

    def test_raises_with_available_list_when_recipe_not_found(
        self, conf_dir: Path, base_cfg: DictConfig
    ) -> None:
        """存在しない recipe を指定したとき利用可能な一覧を含むエラーになること。"""
        cfg = DictConfig(OmegaConf.merge(base_cfg, OmegaConf.create({"recipe": "xgboost"})))
        with pytest.raises(ValueError, match="xgboost"):
            load_recipe_cfgs(cfg, "training", conf_dir)

    def test_raises_custom_empty_dir_message(self, tmp_path: Path, base_cfg: DictConfig) -> None:
        """ディレクトリが空のとき empty_dir_message が使われること。"""
        (tmp_path / "competition" / "titanic" / "training").mkdir(parents=True)
        with pytest.raises(ValueError, match="training 設定が見つかりません"):
            load_recipe_cfgs(
                base_cfg,
                "training",
                tmp_path,
                empty_dir_message="training 設定が見つかりません",
            )

    def test_base_cfg_keys_are_merged(self, conf_dir: Path, base_cfg: DictConfig) -> None:
        """base_cfg のキー（seed など）がマージ結果に含まれること。"""
        cfgs = load_recipe_cfgs(base_cfg, "training", conf_dir)
        assert all(c.seed == 42 for c in cfgs)


class TestLoadSingleRecipeCfg:
    def test_returns_cfg_unchanged_when_recipe_none_and_not_required(
        self, conf_dir: Path, base_cfg: DictConfig
    ) -> None:
        """recipe 未指定・required=False のとき cfg をそのまま返すこと。"""
        result = load_single_recipe_cfg(base_cfg, "training", conf_dir, required=False)
        assert result is base_cfg

    def test_raises_when_recipe_none_and_required(
        self, conf_dir: Path, base_cfg: DictConfig
    ) -> None:
        """recipe 未指定・required=True のとき required_message で ValueError になること。"""
        with pytest.raises(ValueError, match="recipe= が必要です"):
            load_single_recipe_cfg(
                base_cfg,
                "training",
                conf_dir,
                required=True,
                required_message="recipe= が必要です。",
            )

    def test_loads_single_yaml_when_recipe_specified(
        self, conf_dir: Path, base_cfg: DictConfig
    ) -> None:
        """recipe 指定時は該当 yaml をマージすること。"""
        cfg = DictConfig(OmegaConf.merge(base_cfg, OmegaConf.create({"recipe": "lgbm"})))
        result = load_single_recipe_cfg(cfg, "training", conf_dir, required=True)
        assert result.trainer.type == "lgbm"

    def test_raises_when_specified_recipe_not_found(
        self, conf_dir: Path, base_cfg: DictConfig
    ) -> None:
        """存在しない recipe を指定したとき ValueError。"""
        cfg = DictConfig(OmegaConf.merge(base_cfg, OmegaConf.create({"recipe": "nonexistent"})))
        with pytest.raises(ValueError, match="nonexistent"):
            load_single_recipe_cfg(cfg, "training", conf_dir, required=True)
