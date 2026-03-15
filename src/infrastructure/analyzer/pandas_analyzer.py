"""
Pandas ベースの EDA アナライザー。

matplotlib は import 前に非インタラクティブ backend を設定する必要があるため、
モジュール先頭で matplotlib.use("Agg") を呼ぶ。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd
import seaborn as sns
import yaml
from omegaconf import DictConfig, OmegaConf

from src.domain.data.eda import AnalysisStep, EDAResult, FileEDAResult

logger = logging.getLogger(__name__)

_EDA_DIR_GITIGNORE = """\
*
!.gitignore
!*.yaml
!*.md
!*/
"""


class PandasAnalyzer:
    """DataAnalyzer Protocol を満たす Pandas 実装。

    - commit_hash と analyses は DI 層（main.py）から直接渡す。
    - output_format: "parquet" または "csv"（statistics ファイルの保存形式）。
    - 各インスタンスは {report_dir}/{ANALYZER_TYPE}/ サブディレクトリに出力する。
    """

    ANALYZER_TYPE = "pandas"

    def __init__(
        self,
        cfg: DictConfig,
        commit_hash: str,
        analyses: list[AnalysisStep],
        output_format: str = "parquet",
    ) -> None:
        self.cfg = cfg
        self.commit_hash = commit_hash
        self._analyses = analyses
        self.output_format = output_format

    @property
    def _competition_name(self) -> str:
        return str((self.cfg.get("competition") or {}).get("name") or "eda")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> EDAResult:
        self._check_io_separation()
        self._setup_output_gitignore()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        report_dir = (
            Path(self.cfg.output_dir)
            / f"{self._competition_name}_report"
            / timestamp
            / self.ANALYZER_TYPE
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "statistics").mkdir(exist_ok=True)
        (report_dir / "images").mkdir(exist_ok=True)

        csv_files = self._collect_csv_files()

        file_results: list[FileEDAResult] = []
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            file_result = self._analyze_file(df, csv_path, report_dir, self._analyses)
            file_results.append(file_result)

        readme_path = self._write_readme(report_dir, file_results, self._analyses)
        metainfo_path = self._write_metainfo(report_dir, self.commit_hash, csv_files)

        return EDAResult(
            report_dir=report_dir,
            file_results=file_results,
            commit_hash=self.commit_hash,
            readme_path=readme_path,
            metainfo_path=metainfo_path,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _setup_output_gitignore(self) -> None:
        """output_dir に .gitignore を生成し PNG/parquet 等を git 管理外にする。"""
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        gitignore = out / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(_EDA_DIR_GITIGNORE)

    def _check_io_separation(self) -> None:
        """input_paths と output_dir が重複していないことを確認する。

        出力ファイルが入力ディレクトリに混入すると次回実行時の再現性が壊れる。
        """
        out = Path(self.cfg.output_dir).resolve()
        for path_str in self.cfg.input_paths:
            inp = Path(path_str).resolve()
            if out.is_relative_to(inp) or inp.is_relative_to(out):
                raise ValueError(
                    f"output_dir '{out}' と input_path '{inp}' が重複しています。"
                    " 入力と出力は別ディレクトリにしてください。"
                )

    # ------------------------------------------------------------------
    # File collection
    # ------------------------------------------------------------------

    def _collect_csv_files(self) -> list[Path]:
        csv_files: list[Path] = []
        for path_str in self.cfg.input_paths:
            p = Path(path_str)
            if p.is_file():
                csv_files.append(p)
            elif p.is_dir():
                csv_files.extend(sorted(p.glob("*.csv")))
        return csv_files

    # ------------------------------------------------------------------
    # Per-file analysis
    # ------------------------------------------------------------------

    def _analyze_file(
        self,
        df: pd.DataFrame,
        csv_path: Path,
        report_dir: Path,
        analyses: list[AnalysisStep],
    ) -> FileEDAResult:
        stem = csv_path.stem
        output_files: list[Path] = []

        for step in analyses:
            files = self._run_step(step, df, stem, report_dir)
            output_files.extend(files)

        return FileEDAResult(
            source_path=csv_path,
            shape=(len(df), len(df.columns)),
            dtypes={str(col): str(dtype) for col, dtype in df.dtypes.items()},
            missing_counts={
                str(col): int(cnt) for col, cnt in df.isnull().sum().items() if cnt > 0
            },
            output_files=output_files,
        )

    def _run_step(
        self,
        step: AnalysisStep,
        df: pd.DataFrame,
        stem: str,
        report_dir: Path,
    ) -> list[Path]:
        dispatch = {
            "basic_stats": self._step_basic_stats,
            "distributions": self._step_distributions,
            "missing_values": self._step_missing_values,
            "group_stats": self._step_group_stats,
            "id_transitions": self._step_id_transitions,
        }
        if step.type not in dispatch:
            raise ValueError(f"Unknown analysis type: {step.type!r}. Supported: {sorted(dispatch)}")
        return dispatch[step.type](df, stem, report_dir, step.params)

    # ------------------------------------------------------------------
    # Analysis steps
    # ------------------------------------------------------------------

    def _step_basic_stats(
        self,
        df: pd.DataFrame,
        stem: str,
        report_dir: Path,
        _params: dict[str, Any],
    ) -> list[Path]:
        stats_dir = report_dir / "statistics"
        summary = df.describe(include="all").T
        missing = df.isnull().sum().rename("missing_count").to_frame()
        missing["missing_pct"] = missing["missing_count"] / len(df) * 100

        summary_path = self._save_stats(summary, stats_dir / f"{stem}_summary")
        missing_path = self._save_stats(missing, stats_dir / f"{stem}_missing")
        return [summary_path, missing_path]

    def _step_distributions(
        self,
        df: pd.DataFrame,
        stem: str,
        report_dir: Path,
        _params: dict[str, Any],
    ) -> list[Path]:
        images_dir = report_dir / "images"
        max_cols: int = int(self.cfg.max_plot_cols)
        output: list[Path] = []

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        cols = (numeric_cols + cat_cols)[:max_cols]

        for col in cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            if col in numeric_cols:
                ax.hist(df[col].dropna(), bins=30, edgecolor="black")
                ax.set_title(f"{col} distribution")
            else:
                vc = df[col].value_counts().head(20)
                ax.bar(vc.index.astype(str), vc.values.tolist())
                ax.set_title(f"{col} value counts")
                plt.xticks(rotation=45, ha="right")
            ax.set_xlabel(col)
            out = images_dir / f"{stem}_{col}_dist.png"
            fig.tight_layout()
            fig.savefig(out)
            plt.close(fig)
            output.append(out)

        return output

    def _step_missing_values(
        self,
        df: pd.DataFrame,
        stem: str,
        report_dir: Path,
        _params: dict[str, Any],
    ) -> list[Path]:
        images_dir = report_dir / "images"
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_matrix = df.isnull().astype(int)
        sns.heatmap(missing_matrix, cbar=False, ax=ax, yticklabels=False)
        ax.set_title(f"{stem} missing values")
        out = images_dir / f"{stem}_missing_heatmap.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        return [out]

    def _step_group_stats(
        self,
        df: pd.DataFrame,
        stem: str,
        report_dir: Path,
        params: dict[str, Any],
    ) -> list[Path]:
        group_by: str = params["group_by"]
        stats_dir = report_dir / "statistics"
        images_dir = report_dir / "images"
        output: list[Path] = []

        group_stats = df.groupby(group_by).describe(include="all")
        stats_path = self._save_stats(group_stats, stats_dir / f"{stem}_group_stats")
        output.append(stats_path)

        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df[group_by].value_counts()
        ax.bar(counts.index.astype(str), counts.values.tolist())
        ax.set_title(f"{stem} group counts by {group_by}")
        ax.set_xlabel(group_by)
        out = images_dir / f"{stem}_group_counts.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        output.append(out)

        return output

    def _step_id_transitions(
        self,
        df: pd.DataFrame,
        stem: str,
        report_dir: Path,
        params: dict[str, Any],
    ) -> list[Path]:
        id_col: str = params["id_col"]
        images_dir = report_dir / "images"

        if df[id_col].duplicated().sum() == 0:
            return []

        dup_ids = df[id_col][df[id_col].duplicated(keep=False)].unique()
        subset = df[df[id_col].isin(dup_ids)].copy()
        subset["_order"] = range(len(subset))

        fig, ax = plt.subplots(figsize=(8, 5))
        for id_val, grp in subset.groupby(id_col):
            ax.plot(grp["_order"], [str(id_val)] * len(grp), "o-", label=str(id_val))
        ax.set_title(f"{stem} — duplicate ID transitions ({id_col})")
        ax.set_xlabel("row index")
        ax.set_ylabel(id_col)
        out = images_dir / f"{stem}_id_transitions.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        return [out]

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _save_stats(self, df: pd.DataFrame, path_without_ext: Path) -> Path:
        """output_format に応じて .parquet または .csv で保存する。"""
        if self.output_format == "parquet":
            import polars as pl

            out = path_without_ext.with_suffix(".parquet")
            pl.from_pandas(df.reset_index()).write_parquet(out)
        else:
            out = path_without_ext.with_suffix(".csv")
            df.to_csv(out)
        return out

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _write_readme(
        self,
        report_dir: Path,
        file_results: list[FileEDAResult],
        analyses: list[AnalysisStep],
    ) -> Path:
        lines = [
            f"# EDA Report — {self._competition_name} [{self.ANALYZER_TYPE}]",
            "",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "## Files analyzed",
            "",
        ]
        for fr in file_results:
            lines += [
                f"### {fr.source_path.name}",
                "",
                f"Shape: {fr.shape[0]} rows × {fr.shape[1]} cols  |  "
                f"Output files: {len(fr.output_files)}",
                "",
                "#### Schema",
                "",
                "| Column | Type | Missing | Missing % |",
                "|--------|------|--------:|----------:|",
            ]
            for col, dtype in fr.dtypes.items():
                missing = fr.missing_counts.get(col, 0)
                missing_pct = f"{missing / fr.shape[0] * 100:.1f}%" if fr.shape[0] else "-"
                lines.append(f"| {col} | {dtype} | {missing} | {missing_pct} |")
            lines.append("")

        lines += ["## Analyses run", ""]
        for step in analyses:
            lines.append(f"- `{step.type}`" + (f" params={step.params}" if step.params else ""))

        path = report_dir / "README.md"
        path.write_text("\n".join(lines))
        return path

    def _write_metainfo(
        self,
        report_dir: Path,
        commit_hash: str,
        csv_files: list[Path],
    ) -> Path:
        info = {
            "commit_hash": commit_hash,
            "analyzer_type": self.ANALYZER_TYPE,
            "output_format": self.output_format,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "competition": (
                OmegaConf.to_container(self.cfg.competition)
                if self.cfg.get("competition")
                else None
            ),
            "input_files": [str(f) for f in csv_files],
        }
        path = report_dir / "metainfo.yaml"
        path.write_text(yaml.dump(info, allow_unicode=True))
        return path
