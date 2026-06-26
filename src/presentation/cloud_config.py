"""Cloud 設定ヘルパー。

Vertex AI 関連の設定解決ロジックを main.py から分離する。
- ensure_cloud_config: cloud / notification 設定の遅延マージ
- load_trainer_cfgs_safe: pipeline recipe 経由でも安全に trainer config をロード
- resolve_manifest_path: manifest_path の自動解決
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def ensure_cloud_config(cfg: DictConfig, conf_dir: str | None = None) -> DictConfig:
    """cloud 設定が未解決の場合、conf/cloud/vertex.yaml を手動マージする。

    Pipeline 経由の場合、Hydra の defaults 処理が走らないため
    cloud: null のままになることがある。その場合は明示的にロードしてマージする。

    同様に notification 設定も未解決なら conf/notification/slack.yaml をマージする。
    """
    if conf_dir is None:
        conf_dir = str(Path(__file__).parent.parent.parent / "conf")
    needs_merge = False
    extras: list[object] = []

    if cfg.get("cloud") is None:
        cloud_yaml = Path(conf_dir) / "cloud" / "vertex.yaml"
        if not cloud_yaml.exists():
            raise FileNotFoundError(f"Cloud config not found: {cloud_yaml}")
        extras.append(OmegaConf.load(cloud_yaml))
        needs_merge = True

    if cfg.get("notification") is None:
        slack_yaml = Path(conf_dir) / "notification" / "slack.yaml"
        if slack_yaml.exists():
            extras.append(OmegaConf.load(slack_yaml))
            needs_merge = True

    if not needs_merge:
        return cfg

    base = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for extra in extras:
        base = OmegaConf.merge(base, extra)
    return DictConfig(base)


def load_trainer_cfgs_safe(cfg: DictConfig, conf_dir: str | None = None) -> list[DictConfig]:
    """recipe が pipeline recipe の場合でも安全に trainer config をロードする。

    pipeline 経由だと cfg.recipe が pipeline recipe（例: cloud_download_and_push）に
    なっており、trainer yaml として解決できない。recipe を一時的に null にして
    全 trainer をロードする。
    """
    from src.usecase.training.trainer_loader import load_trainer_cfgs

    if conf_dir is None:
        conf_dir = str(Path(__file__).parent.parent.parent / "conf")
    cfg_copy = DictConfig(OmegaConf.to_container(cfg, resolve=True))
    cfg_copy.recipe = None
    return load_trainer_cfgs(cfg_copy, Path(conf_dir))


def resolve_manifest_path(cfg: DictConfig, conf_dir: str | None = None) -> Path:
    """manifest_path を解決する。

    manifest_path が指定されていればそのまま返す。
    未指定の場合は competition + job_id + latest から自動解決する。
    job_id が未設定なら recipe から trainer config をロードして取得する。
    """
    from src.usecase._utils import resolve_latest_dir

    if conf_dir is None:
        conf_dir = str(Path(__file__).parent.parent.parent / "conf")

    explicit = cfg.get("manifest_path")
    if explicit is not None and str(explicit) != "None":
        return Path(str(explicit))

    competition = str(cfg.competition.name)
    history_base = str(cfg.get("cloud_jobs_history_dir", "cloud_jobs_history"))

    # job_id を解決: cfg の job_id が使えなければ trainer config から取得
    # pipeline 経由だと cfg.job_id が pipeline 自体の ID になるため、
    # 実際の history ディレクトリが存在するかで判定する
    job_id = cfg.get("job_id")
    if job_id is not None and str(job_id) != "None":
        candidate = Path(history_base) / competition / str(job_id)
        if candidate.is_dir():
            latest_dir = resolve_latest_dir(f"{candidate}/latest")
            return Path(latest_dir) / "job_manifest.yaml"

    # cfg.job_id が無い or 対応 dir が無い → trainer config から取得
    trainer_cfgs = load_trainer_cfgs_safe(cfg, conf_dir)
    job_id = str(trainer_cfgs[0].job_id)
    latest_dir = resolve_latest_dir(f"{history_base}/{competition}/{job_id}/latest")
    return Path(latest_dir) / "job_manifest.yaml"
