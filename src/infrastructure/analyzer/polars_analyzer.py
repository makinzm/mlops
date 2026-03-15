"""
Polars ベースの EDA アナライザー。

データの読み込みと統計計算を polars で行い、統計ファイルは常に .parquet で保存する。
プロット生成には matplotlib を使用（polars Series → list 変換経由）。

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
import polars as pl
import seaborn as sns
import yaml
from omegaconf import DictConfig, OmegaConf

from src.domain.data.eda import AnalysisStep, EDAResult, FileEDAResult

logger = logging.getLogger(__name__)


class PolarsAnalyzer:
    """DataAnalyzer Protocol を満たす Polars 実装。

    - commit_hash と analyses は DI 層（main.py）から直接渡す。
    - 統計ファイルは常に .parquet（polars ネイティブ）で保存する。
    - 各インスタンスは {report_dir}/{ANALYZER_TYPE}/ サブディレクトリに出力する。
    """

    ANALYZER_TYPE = "polars"

    def __init__(
        self,
        cfg: DictConfig,
        commit_hash: str,
        analyses: list[AnalysisStep],
    ) -> None:
        self.cfg = cfg
        self.commit_hash = commit_hash
        self._analyses = analyses

    @property
    def _competition_name(self) -> str:
        return str((self.cfg.get("competition") or {}).get("name") or "eda")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> EDAResult:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        report_dir = (
            Path(self.cfg.report_dir)
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
            df = pl.read_csv(csv_path, infer_schema_length=10000)
            file_result = self._analyze_file(df, csv_path, report_dir, self._analyses)
            file_results.append(file_result)

        readme_path = self._write_readme(report_dir, file_results, self._analyses)
        metainfo_path = self._write_metainfo(report_dir, csv_files)

        return EDAResult(
            report_dir=report_dir,
            file_results=file_results,
            commit_hash=self.commit_hash,
            readme_path=readme_path,
            metainfo_path=metainfo_path,
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
        df: pl.DataFrame,
        csv_path: Path,
        report_dir: Path,
        analyses: list[AnalysisStep],
    ) -> FileEDAResult:
        stem = csv_path.stem
        output_files: list[Path] = []

        for step in analyses:
            files = self._run_step(step, df, stem, report_dir)
            output_files.extend(files)

        null_counts = df.null_count().row(0)
        return FileEDAResult(
            source_path=csv_path,
            shape=(df.height, df.width),
            dtypes={col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            missing_counts={col: cnt for col, cnt in zip(df.columns, null_counts) if cnt > 0},
            output_files=output_files,
        )

    def _run_step(
        self,
        step: AnalysisStep,
        df: pl.DataFrame,
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
        df: pl.DataFrame,
        stem: str,
        report_dir: Path,
        _params: dict[str, Any],
    ) -> list[Path]:
        stats_dir = report_dir / "statistics"
        summary = df.describe()
        null_counts = df.null_count()
        missing = pl.DataFrame(
            {
                "column": df.columns,
                "missing_count": list(null_counts.row(0)),
                "missing_pct": [
                    cnt / df.height * 100 if df.height else 0.0 for cnt in null_counts.row(0)
                ],
            }
        )
        summary_path = stats_dir / f"{stem}_summary.parquet"
        missing_path = stats_dir / f"{stem}_missing.parquet"
        summary.write_parquet(summary_path)
        missing.write_parquet(missing_path)
        return [summary_path, missing_path]

    def _step_distributions(
        self,
        df: pl.DataFrame,
        stem: str,
        report_dir: Path,
        _params: dict[str, Any],
    ) -> list[Path]:
        images_dir = report_dir / "images"
        max_cols: int = int(self.cfg.max_plot_cols)
        output: list[Path] = []

        numeric_cols = [c for c, t in zip(df.columns, df.dtypes) if t.is_numeric()]
        cat_cols = [c for c, t in zip(df.columns, df.dtypes) if not t.is_numeric()]
        cols = (numeric_cols + cat_cols)[:max_cols]

        for col in cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            if col in numeric_cols:
                data = df[col].drop_nulls().to_list()
                ax.hist(data, bins=30, edgecolor="black")
                ax.set_title(f"{col} distribution")
            else:
                vc = df[col].value_counts().sort("count", descending=True).head(20)
                ax.bar(vc[col].to_list(), vc["count"].to_list())
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
        df: pl.DataFrame,
        stem: str,
        report_dir: Path,
        _params: dict[str, Any],
    ) -> list[Path]:
        images_dir = report_dir / "images"
        null_matrix = df.select([pl.col(c).is_null().cast(pl.Int32) for c in df.columns]).to_numpy()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(null_matrix, cbar=False, ax=ax, yticklabels=False)
        ax.set_title(f"{stem} missing values")
        out = images_dir / f"{stem}_missing_heatmap.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        return [out]

    def _step_group_stats(
        self,
        df: pl.DataFrame,
        stem: str,
        report_dir: Path,
        params: dict[str, Any],
    ) -> list[Path]:
        group_by: str = params["group_by"]
        stats_dir = report_dir / "statistics"
        images_dir = report_dir / "images"
        output: list[Path] = []

        numeric_cols = [
            c for c, t in zip(df.columns, df.dtypes) if t.is_numeric() and c != group_by
        ]
        aggs = [pl.col(c).mean().alias(f"{c}_mean") for c in numeric_cols]
        aggs += [pl.col(c).std().alias(f"{c}_std") for c in numeric_cols]
        aggs.append(pl.len().alias("count"))
        group_stats = df.group_by(group_by).agg(aggs).sort(group_by)

        stats_path = stats_dir / f"{stem}_group_stats.parquet"
        group_stats.write_parquet(stats_path)
        output.append(stats_path)

        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df[group_by].value_counts().sort(group_by)
        ax.bar(
            [str(v) for v in counts[group_by].to_list()],
            counts["count"].to_list(),
        )
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
        df: pl.DataFrame,
        stem: str,
        report_dir: Path,
        params: dict[str, Any],
    ) -> list[Path]:
        id_col: str = params["id_col"]
        images_dir = report_dir / "images"

        if not df[id_col].is_duplicated().any():
            return []

        dup_ids = df.filter(pl.col(id_col).is_duplicated())[id_col].unique().to_list()
        subset = df.filter(pl.col(id_col).is_in(dup_ids)).with_row_index("_order")

        fig, ax = plt.subplots(figsize=(8, 5))
        for id_val in dup_ids:
            grp = subset.filter(pl.col(id_col) == id_val)
            ax.plot(
                grp["_order"].to_list(),
                [str(id_val)] * grp.height,
                "o-",
                label=str(id_val),
            )
        ax.set_title(f"{stem} — duplicate ID transitions ({id_col})")
        ax.set_xlabel("row index")
        ax.set_ylabel(id_col)
        out = images_dir / f"{stem}_id_transitions.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        return [out]

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

    def _write_metainfo(self, report_dir: Path, csv_files: list[Path]) -> Path:
        info = {
            "commit_hash": self.commit_hash,
            "analyzer_type": self.ANALYZER_TYPE,
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
