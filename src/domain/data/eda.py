"""
EDA ドメイン定義。

EDAResult / FileEDAResult / AnalysisStep の dataclass と
DataAnalyzer Protocol を定義する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class AnalysisStep:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class FileEDAResult:
    source_path: Path
    shape: tuple[int, int]
    dtypes: dict[str, str]
    missing_counts: dict[str, int]
    output_files: list[Path]


@dataclass
class EDAResult:
    report_dir: Path
    file_results: list[FileEDAResult]
    commit_hash: str
    readme_path: Path
    metainfo_path: Path


@runtime_checkable
class DataAnalyzer(Protocol):
    def analyze(self) -> EDAResult: ...
