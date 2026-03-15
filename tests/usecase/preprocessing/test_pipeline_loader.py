"""
preprocess パイプライン検出ロジックのテスト。

なぜこのテストが必要か:
- `usecase=preprocess` は competition の conf/preprocess/ 配下を自動検出して実行する。
- 検出・マージ・エラー通知のロジックを UseCase や main.py から分離し、
  Hydra なしで単体テストできるようにする。
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.usecase.preprocessing.pipeline_loader import load_pipeline_cfgs


@pytest.fixture()
def conf_dir(tmp_path: Path) -> Path:
    """titanic/preprocess/ に base.yaml と lgbm.yaml を持つ偽 conf ツリーを返す。"""
    preprocess_dir = tmp_path / "competition" / "titanic" / "preprocess"
    preprocess_dir.mkdir(parents=True)
    (preprocess_dir / "base.yaml").write_text("job_id: base_job\noutput_dir: out/base\n")
    (preprocess_dir / "lgbm.yaml").write_text("job_id: lgbm_job\noutput_dir: out/lgbm\n")
    return tmp_path


class TestLoadAllPipelines:
    def test_returns_all_yamls_when_no_pipeline_specified(self, conf_dir: Path) -> None:
        """pipeline を指定しないとき preprocess/ 配下の全 yaml がロードされる。

        ensemble 時に base / lgbm / nn など複数のパイプラインを一括実行するため。
        """
        cfg = OmegaConf.create({"competition": {"name": "titanic"}})
        cfgs = load_pipeline_cfgs(cfg, conf_dir)
        assert len(cfgs) == 2

    def test_configs_are_merged_with_base_cfg(self, conf_dir: Path) -> None:
        """ロードした各 config には元の cfg（competition 等）がマージされている。"""
        cfg = OmegaConf.create({"competition": {"name": "titanic"}, "seed": 42})
        cfgs = load_pipeline_cfgs(cfg, conf_dir)
        assert all(c.seed == 42 for c in cfgs)

    def test_pipeline_specific_key_is_present(self, conf_dir: Path) -> None:
        """各 config に pipeline yaml 固有のキー（job_id）が含まれる。"""
        cfg = OmegaConf.create({"competition": {"name": "titanic"}})
        cfgs = load_pipeline_cfgs(cfg, conf_dir)
        job_ids = {c.job_id for c in cfgs}
        assert job_ids == {"base_job", "lgbm_job"}


class TestLoadSpecificPipeline:
    def test_returns_only_specified_pipeline(self, conf_dir: Path) -> None:
        """pipeline=base のとき base.yaml だけがロードされる。"""
        cfg = OmegaConf.create({"competition": {"name": "titanic"}, "pipeline": "base"})
        cfgs = load_pipeline_cfgs(cfg, conf_dir)
        assert len(cfgs) == 1
        assert cfgs[0].job_id == "base_job"

    def test_raises_with_available_list_when_pipeline_not_found(self, conf_dir: Path) -> None:
        """存在しない pipeline を指定したとき利用可能な一覧を含むエラーが発生する。

        ユーザーが正確な名前を調べなくても次のアクションがわかるようにするため。
        """
        cfg = OmegaConf.create({"competition": {"name": "titanic"}, "pipeline": "nonexistent"})
        with pytest.raises(ValueError, match="nonexistent"):
            load_pipeline_cfgs(cfg, conf_dir)

    def test_error_message_lists_available_pipelines(self, conf_dir: Path) -> None:
        """エラーメッセージに利用可能なパイプライン名が含まれる。"""
        cfg = OmegaConf.create({"competition": {"name": "titanic"}, "pipeline": "missing"})
        with pytest.raises(ValueError, match="base"):
            load_pipeline_cfgs(cfg, conf_dir)


class TestEmptyPreprocessDir:
    def test_raises_when_no_yaml_files_exist(self, tmp_path: Path) -> None:
        """preprocess/ が空のとき分かりやすいエラーが発生する。"""
        preprocess_dir = tmp_path / "competition" / "empty-comp" / "preprocess"
        preprocess_dir.mkdir(parents=True)
        cfg = OmegaConf.create({"competition": {"name": "empty-comp"}})
        with pytest.raises(ValueError, match="前処理設定が見つかりません"):
            load_pipeline_cfgs(cfg, tmp_path)
