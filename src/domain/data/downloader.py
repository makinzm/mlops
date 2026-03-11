"""
データダウンロードのドメイン定義。

DataDownloader Protocol と DownloadResult を定義する。
インフラ実装はこの Protocol を満たすことで UseCase から利用可能になる。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class DownloadResult:
    output_dir: Path
    files: list[Path]
    commit_hash: str  # DoD: 全てのコードは Git の CommitHash が記録されること


@runtime_checkable
class DataDownloader(Protocol):
    def download(self) -> DownloadResult: ...
