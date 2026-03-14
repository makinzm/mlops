"""
PandasAnalyzer のインフラテスト。

実際にファイルを生成して出力内容を検証する。
tmp_path フィクスチャでテンポラリディレクトリを使い、
テスト後に自動クリーンアップされる。
"""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.infrastructure.analyzer.pandas_analyzer import PandasAnalyzer

# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------


def _make_cfg(
    tmp_path: Path,
    csv_path: Path,
    analyses: list[dict],  # type: ignore[type-arg]
    output_format: str = "polars",
    max_plot_cols: int = 20,
) -> object:
    """テスト用 DictConfig を生成する。"""
    return OmegaConf.create(
        {
            "seed": 42,
            "report_dir": str(tmp_path / "competition"),
            "output_format": output_format,
            "max_plot_cols": max_plot_cols,
            "competition": {
                "name": "titanic",
                "input_paths": [str(csv_path)],
            },
            "analyses": analyses,
        }
    )


def _write_titanic_csv(path: Path) -> Path:
    """最小限の Titanic 風 CSV をテスト用に生成する。"""
    df = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4, 5],
            "Survived": [0, 1, 1, 0, 1],
            "Pclass": [3, 1, 3, 1, 3],
            "Name": ["A", "B", "C", "D", "E"],
            "Age": [22.0, 38.0, None, 35.0, 35.0],
            "Fare": [7.25, 71.28, 7.92, 53.1, 8.05],
        }
    )
    csv_file = path / "train.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture()
def titanic_csv(tmp_path: Path) -> Path:
    return _write_titanic_csv(tmp_path)


def _make_analyzer(cfg: object) -> PandasAnalyzer:
    git_repo = MagicMock()
    git_repo.get_commit_hash.return_value = "deadbeef"
    return PandasAnalyzer(cfg, git_repo)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# レポートディレクトリ生成
# ---------------------------------------------------------------------------


class TestReportDirectoryCreation:
    def test_report_dir_created(self, titanic_csv: Path, tmp_path: Path) -> None:
        """analyze() 呼び出しでレポートディレクトリが作成される。"""
        cfg = _make_cfg(tmp_path, titanic_csv, [{"type": "basic_stats"}])
        analyzer = _make_analyzer(cfg)

        result = analyzer.analyze()

        assert result.report_dir.exists()

    def test_gitignore_contains_star(self, titanic_csv: Path, tmp_path: Path) -> None:
        """.gitignore が '*' のみ含み、レポート内ファイルを git 対象外にする。"""
        cfg = _make_cfg(tmp_path, titanic_csv, [{"type": "basic_stats"}])
        analyzer = _make_analyzer(cfg)

        result = analyzer.analyze()

        gitignore = result.report_dir / ".gitignore"
        assert gitignore.exists()
        assert gitignore.read_text().strip() == "*"


# ---------------------------------------------------------------------------
# basic_stats
# ---------------------------------------------------------------------------


class TestBasicStats:
    def test_basic_stats_creates_summary(self, titanic_csv: Path, tmp_path: Path) -> None:
        """basic_stats ステップで summary ファイルが statistics/ 下に生成される。"""
        cfg = _make_cfg(tmp_path, titanic_csv, [{"type": "basic_stats"}])
        analyzer = _make_analyzer(cfg)

        result = analyzer.analyze()

        stats_dir = result.report_dir / "statistics"
        summary_files = list(stats_dir.glob("train_summary*"))
        assert len(summary_files) == 1

    def test_basic_stats_creates_missing(self, titanic_csv: Path, tmp_path: Path) -> None:
        """basic_stats で欠損値集計ファイルが生成される。"""
        cfg = _make_cfg(tmp_path, titanic_csv, [{"type": "basic_stats"}])
        analyzer = _make_analyzer(cfg)

        result = analyzer.analyze()

        stats_dir = result.report_dir / "statistics"
        missing_files = list(stats_dir.glob("train_missing*"))
        assert len(missing_files) == 1


# ---------------------------------------------------------------------------
# distributions
# ---------------------------------------------------------------------------


class TestDistributions:
    def test_distributions_creates_images(self, titanic_csv: Path, tmp_path: Path) -> None:
        """distributions ステップで数値・カテゴリ列の PNG が images/ に生成される。"""
        cfg = _make_cfg(tmp_path, titanic_csv, [{"type": "distributions"}])
        analyzer = _make_analyzer(cfg)

        result = analyzer.analyze()

        images_dir = result.report_dir / "images"
        pngs = list(images_dir.glob("train_*_dist.png"))
        assert len(pngs) >= 1

    def test_max_plot_cols_limits_images(self, tmp_path: Path) -> None:
        """max_plot_cols=2 のとき distributions が 2 件以内の PNG を生成する。"""
        # 列が多い CSV を生成
        wide_csv = tmp_path / "wide.csv"
        df = pd.DataFrame({f"col{i}": range(5) for i in range(10)})
        df.to_csv(wide_csv, index=False)

        cfg = _make_cfg(
            tmp_path,
            wide_csv,
            [{"type": "distributions"}],
            max_plot_cols=2,
        )
        analyzer = _make_analyzer(cfg)

        result = analyzer.analyze()

        images_dir = result.report_dir / "images"
        pngs = list(images_dir.glob("wide_*_dist.png"))
        assert len(pngs) <= 2


# ---------------------------------------------------------------------------
# missing_values
# ---------------------------------------------------------------------------


class TestMissingValues:
    def test_missing_values_creates_heatmap(self, titanic_csv: Path, tmp_path: Path) -> None:
        """missing_values ステップで欠損値ヒートマップ PNG が生成される。"""
        cfg = _make_cfg(tmp_path, titanic_csv, [{"type": "missing_values"}])
        analyzer = _make_analyzer(cfg)

        result = analyzer.analyze()

        images_dir = result.report_dir / "images"
        heatmaps = list(images_dir.glob("train_missing_heatmap.png"))
        assert len(heatmaps) == 1


# ---------------------------------------------------------------------------
# group_stats
# ---------------------------------------------------------------------------


class TestGroupStats:
    def test_group_stats_creates_file_and_image(self, titanic_csv: Path, tmp_path: Path) -> None:
        """group_stats ステップで統計ファイルとグループ棒グラフが生成される。"""
        cfg = _make_cfg(
            tmp_path,
            titanic_csv,
            [{"type": "group_stats", "group_by": "Survived"}],
        )
        analyzer = _make_analyzer(cfg)

        result = analyzer.analyze()

        stats_dir = result.report_dir / "statistics"
        images_dir = result.report_dir / "images"
        assert len(list(stats_dir.glob("train_group_stats*"))) == 1
        assert len(list(images_dir.glob("train_group_counts.png"))) == 1

    def test_group_stats_not_run_when_not_in_analyses(
        self, titanic_csv: Path, tmp_path: Path
    ) -> None:
        """analyses リストに group_stats がなければ group_stats ファイルは生成されない。"""
        cfg = _make_cfg(tmp_path, titanic_csv, [{"type": "basic_stats"}])
        analyzer = _make_analyzer(cfg)

        result = analyzer.analyze()

        stats_dir = result.report_dir / "statistics"
        assert len(list(stats_dir.glob("*group_stats*"))) == 0


# ---------------------------------------------------------------------------
# id_transitions
# ---------------------------------------------------------------------------


class TestIdTransitions:
    def test_id_transitions_image_for_duplicate_ids(self, tmp_path: Path) -> None:
        """重複 ID があるとき id_transitions 画像が生成される。"""
        csv_file = tmp_path / "dup.csv"
        df = pd.DataFrame({"id": [1, 1, 2, 3], "value": [10, 20, 30, 40]})
        df.to_csv(csv_file, index=False)

        cfg = _make_cfg(
            tmp_path,
            csv_file,
            [{"type": "id_transitions", "id_col": "id"}],
        )
        analyzer = _make_analyzer(cfg)

        result = analyzer.analyze()

        images_dir = result.report_dir / "images"
        assert len(list(images_dir.glob("dup_id_transitions.png"))) == 1

    def test_no_id_transitions_when_no_duplicates(self, tmp_path: Path) -> None:
        """重複 ID がないとき id_transitions 画像は生成されない。"""
        csv_file = tmp_path / "nodup.csv"
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        df.to_csv(csv_file, index=False)

        cfg = _make_cfg(
            tmp_path,
            csv_file,
            [{"type": "id_transitions", "id_col": "id"}],
        )
        analyzer = _make_analyzer(cfg)

        result = analyzer.analyze()

        images_dir = result.report_dir / "images"
        assert len(list(images_dir.glob("nodup_id_transitions.png"))) == 0


# ---------------------------------------------------------------------------
# 出力フォーマット
# ---------------------------------------------------------------------------


class TestOutputFormat:
    def test_polars_output_saves_parquet(self, titanic_csv: Path, tmp_path: Path) -> None:
        """output_format=polars のとき statistics が .parquet で保存される。"""
        cfg = _make_cfg(tmp_path, titanic_csv, [{"type": "basic_stats"}], output_format="polars")
        analyzer = _make_analyzer(cfg)

        result = analyzer.analyze()

        stats_dir = result.report_dir / "statistics"
        assert len(list(stats_dir.glob("*.parquet"))) >= 1
        assert len(list(stats_dir.glob("*.csv"))) == 0

    def test_pandas_output_saves_csv(self, titanic_csv: Path, tmp_path: Path) -> None:
        """output_format=pandas のとき statistics が .csv で保存される。"""
        cfg = _make_cfg(tmp_path, titanic_csv, [{"type": "basic_stats"}], output_format="pandas")
        analyzer = _make_analyzer(cfg)

        result = analyzer.analyze()

        stats_dir = result.report_dir / "statistics"
        assert len(list(stats_dir.glob("*.csv"))) >= 1
        assert len(list(stats_dir.glob("*.parquet"))) == 0


# ---------------------------------------------------------------------------
# metainfo
# ---------------------------------------------------------------------------


class TestMetainfo:
    def test_metainfo_contains_commit_hash(self, titanic_csv: Path, tmp_path: Path) -> None:
        """metainfo.yaml に commit_hash が含まれること（DoD 要件）。"""
        cfg = _make_cfg(tmp_path, titanic_csv, [{"type": "basic_stats"}])
        analyzer = _make_analyzer(cfg)

        result = analyzer.analyze()

        assert result.metainfo_path.exists()
        content = result.metainfo_path.read_text()
        assert "deadbeef" in content


# ---------------------------------------------------------------------------
# input_paths の指定形式
# ---------------------------------------------------------------------------


class TestInputPaths:
    def test_input_path_as_file(self, titanic_csv: Path, tmp_path: Path) -> None:
        """input_paths にファイルを直接指定したとき分析が実行される。"""
        cfg = _make_cfg(tmp_path, titanic_csv, [{"type": "basic_stats"}])
        analyzer = _make_analyzer(cfg)

        result = analyzer.analyze()

        assert len(result.file_results) == 1

    def test_input_path_as_directory(self, tmp_path: Path) -> None:
        """input_paths にディレクトリを指定したとき CSV が収集されて分析される。"""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        _write_titanic_csv(raw_dir)

        cfg = OmegaConf.create(
            {
                "seed": 42,
                "report_dir": str(tmp_path / "competition"),
                "output_format": "pandas",
                "max_plot_cols": 20,
                "competition": {
                    "name": "titanic",
                    "input_paths": [str(raw_dir)],
                },
                "analyses": [{"type": "basic_stats"}],
            }
        )
        analyzer = _make_analyzer(cfg)

        result = analyzer.analyze()

        assert len(result.file_results) >= 1


# ---------------------------------------------------------------------------
# エラーハンドリング
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_unknown_analysis_type_raises(self, titanic_csv: Path, tmp_path: Path) -> None:
        """未知の analysis type は ValueError を送出する。"""
        cfg = _make_cfg(tmp_path, titanic_csv, [{"type": "nonexistent_step"}])
        analyzer = _make_analyzer(cfg)

        with pytest.raises(ValueError, match="Unknown analysis type"):
            analyzer.analyze()
