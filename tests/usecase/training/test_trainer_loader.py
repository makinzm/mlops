"""
Phase 2: trainer_loader — 設定から Trainer を選択するファクトリのテスト。

なぜこのテストが必要か:
  - `trainer_loader.load_trainer_cfgs()` は conf/competition/{name}/training/*.yaml を
    検出し Hydra cfg とマージした設定リストを返す。preprocess の pipeline_loader と
    同じパターンを training 用に再現する。
  - `trainer_loader.resolve_trainer()` は cfg.trainer.type に基づいて
    Trainer Protocol の実装を返すファクトリ。新しい trainer 追加時に
    ここだけ変更すれば良い設計を保証する。
  - trainer が未登録の type を渡されたときに明確な ValueError が出ることを確認する。
"""

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from src.domain.model.trainer import Trainer
from src.usecase.training.trainer_loader import load_trainer_cfgs, resolve_trainer

# ──────────────────────────────────────────────────────────────
# load_trainer_cfgs
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def conf_dir(tmp_path: Path) -> Path:
    """competition/titanic/training/ に yaml ファイルを持つ仮の conf ディレクトリ。"""
    training_dir = tmp_path / "competition" / "titanic" / "training"
    training_dir.mkdir(parents=True)

    (training_dir / "lgbm.yaml").write_text("trainer:\n  type: lgbm\n  n_folds: 5\n  seed: 42\n")
    (training_dir / "nn.yaml").write_text("trainer:\n  type: nn\n  n_folds: 5\n  seed: 42\n")
    return tmp_path


@pytest.fixture
def base_cfg() -> DictConfig:
    return OmegaConf.create({"competition": {"name": "titanic"}, "seed": 42})


class TestLoadTrainerCfgs:
    def test_loads_all_yamls_when_no_trainer_specified(
        self, conf_dir: Path, base_cfg: DictConfig
    ) -> None:
        """trainer 未指定時は全ての yaml をロードすること。"""
        cfgs = load_trainer_cfgs(base_cfg, conf_dir)
        types = [c.trainer.type for c in cfgs]
        assert sorted(types) == ["lgbm", "nn"]

    def test_loads_single_yaml_when_trainer_specified(
        self, conf_dir: Path, base_cfg: DictConfig
    ) -> None:
        """trainer=lgbm を指定したときは lgbm.yaml のみロードすること。"""
        cfg = DictConfig(OmegaConf.merge(base_cfg, OmegaConf.create({"trainer_name": "lgbm"})))
        cfgs = load_trainer_cfgs(cfg, conf_dir)
        assert len(cfgs) == 1
        assert cfgs[0].trainer.type == "lgbm"

    def test_raises_when_specified_trainer_not_found(
        self, conf_dir: Path, base_cfg: DictConfig
    ) -> None:
        """存在しない trainer 名を指定したときは ValueError。"""
        cfg = DictConfig(OmegaConf.merge(base_cfg, OmegaConf.create({"trainer_name": "xgboost"})))
        with pytest.raises(ValueError, match="xgboost"):
            load_trainer_cfgs(cfg, conf_dir)

    def test_raises_when_training_dir_is_empty(self, tmp_path: Path, base_cfg: DictConfig) -> None:
        """training ディレクトリが空のとき ValueError。"""
        (tmp_path / "competition" / "titanic" / "training").mkdir(parents=True)
        with pytest.raises(ValueError, match="training"):
            load_trainer_cfgs(base_cfg, tmp_path)

    def test_base_cfg_keys_are_merged(self, conf_dir: Path, base_cfg: DictConfig) -> None:
        """base_cfg のキー（seed など）がマージ結果に含まれること。"""
        cfgs = load_trainer_cfgs(base_cfg, conf_dir)
        assert all(c.seed == 42 for c in cfgs)


# ──────────────────────────────────────────────────────────────
# resolve_trainer
# ──────────────────────────────────────────────────────────────


class TestResolveTrainer:
    def test_returns_trainer_protocol_instance(self) -> None:
        """resolve_trainer は Trainer Protocol を満たすオブジェクトを返すこと。"""
        cfg = OmegaConf.create(
            {
                "trainer": {"type": "lgbm", "n_folds": 5, "seed": 42},
                "competition": {"name": "titanic"},
            }
        )
        trainer = resolve_trainer(cfg)
        assert isinstance(trainer, Trainer)

    def test_raises_for_unknown_trainer_type(self) -> None:
        """未登録の trainer type は ValueError。"""
        cfg = OmegaConf.create(
            {
                "trainer": {"type": "catboost", "n_folds": 5, "seed": 42},
                "competition": {"name": "titanic"},
            }
        )
        with pytest.raises(ValueError, match="catboost"):
            resolve_trainer(cfg)
