"""
inference_loader のテスト。

なぜこのテストが必要か:
  - load_inference_cfgs は conf/competition/{name}/inference/*.yaml を検出し
    cfg とマージするが、これまでテストが存在せず回帰検知できなかった。
  - trainer_loader / preprocessing.pipeline_loader と同じ recipe ロードパターンを
    共通実装（src/usecase/_recipe.py）に委譲しているため、その委譲が正しく
    機能することをここで固定する。
"""

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from src.usecase.inference.inference_loader import load_inference_cfgs


@pytest.fixture
def conf_dir(tmp_path: Path) -> Path:
    """competition/titanic/inference/ に yaml ファイルを持つ仮の conf ディレクトリ。"""
    inference_dir = tmp_path / "competition" / "titanic" / "inference"
    inference_dir.mkdir(parents=True)
    (inference_dir / "lgbm.yaml").write_text("model: lgbm\n")
    (inference_dir / "ensemble.yaml").write_text("model: ensemble\n")
    return tmp_path


@pytest.fixture
def base_cfg() -> DictConfig:
    return OmegaConf.create({"competition": {"name": "titanic"}, "seed": 42})


class TestLoadInferenceCfgs:
    def test_loads_all_yamls_when_no_recipe_specified(
        self, conf_dir: Path, base_cfg: DictConfig
    ) -> None:
        """recipe 未指定時は inference/ 配下の全 yaml をロードすること。"""
        cfgs = load_inference_cfgs(base_cfg, conf_dir)
        models = sorted(c.model for c in cfgs)
        assert models == ["ensemble", "lgbm"]

    def test_loads_single_yaml_when_recipe_specified(
        self, conf_dir: Path, base_cfg: DictConfig
    ) -> None:
        """recipe=lgbm を指定したときは lgbm.yaml のみロードすること。"""
        cfg = DictConfig(OmegaConf.merge(base_cfg, OmegaConf.create({"recipe": "lgbm"})))
        cfgs = load_inference_cfgs(cfg, conf_dir)
        assert len(cfgs) == 1
        assert cfgs[0].model == "lgbm"

    def test_raises_when_specified_recipe_not_found(
        self, conf_dir: Path, base_cfg: DictConfig
    ) -> None:
        """存在しない recipe を指定したとき ValueError。"""
        cfg = DictConfig(OmegaConf.merge(base_cfg, OmegaConf.create({"recipe": "nonexistent"})))
        with pytest.raises(ValueError, match="nonexistent"):
            load_inference_cfgs(cfg, conf_dir)

    def test_raises_when_inference_dir_is_empty(self, tmp_path: Path, base_cfg: DictConfig) -> None:
        """inference ディレクトリが空のとき ValueError。"""
        (tmp_path / "competition" / "titanic" / "inference").mkdir(parents=True)
        with pytest.raises(ValueError, match="inference"):
            load_inference_cfgs(base_cfg, tmp_path)

    def test_base_cfg_keys_are_merged(self, conf_dir: Path, base_cfg: DictConfig) -> None:
        """base_cfg のキー（seed など）がマージ結果に含まれること。"""
        cfgs = load_inference_cfgs(base_cfg, conf_dir)
        assert all(c.seed == 42 for c in cfgs)
