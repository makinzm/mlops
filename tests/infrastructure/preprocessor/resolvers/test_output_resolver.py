"""
OutputResolver のテスト。

なぜこのテストが必要か:
- OutputResolver は学習コードが読むファイルを生成する最終ステップ。
- cv=false → {node_id}/test.parquet（サブディレクトリ）
  cv=true  → {node_id}/fold_N/train.parquet + fold_N/test.parquet
  という出力形式の差分は学習コードと推論コードの読み込み方に直結するため厳密に検証する。
- cv=false をサブディレクトリ形式にするのは InferenceUseCase が
  {preprocess_output_dir}/test_out/test.parquet を期待するため。
- CV 分割が「fold ごとに Train/Test が正しく分かれている」ことを確認する。
"""

from pathlib import Path

import polars as pl
import pytest

from src.infrastructure.preprocessor.resolvers.output_resolver import OutputResolver


@pytest.fixture()
def resolver() -> OutputResolver:
    return OutputResolver()


@pytest.fixture()
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": list(range(10)),
            "col1": [float(i) for i in range(10)],
            "label": [i % 2 for i in range(10)],
        }
    )


class TestOutputNoCV:
    def test_subdir_parquet_created(
        self, resolver: OutputResolver, sample_df: pl.DataFrame, tmp_path: Path
    ) -> None:
        """cv=false の場合、{node_id}/test.parquet がサブディレクトリに生成されること。

        なぜこの形式か:
        InferenceUseCase が {preprocess_output_dir}/test_out/test.parquet を期待するため。
        フラットファイル（test_out.parquet）では inference が FileNotFoundError になる。
        """
        resolver.output(
            df=sample_df,
            output_dir=tmp_path,
            node_id="tabular_out",
            columns=["id", "col1", "label"],
            cv=False,
            splits=None,
        )
        assert (tmp_path / "tabular_out" / "test.parquet").exists()

    def test_parquet_content_matches(
        self, resolver: OutputResolver, sample_df: pl.DataFrame, tmp_path: Path
    ) -> None:
        """生成された parquet の内容が指定カラムに一致すること。"""
        resolver.output(
            df=sample_df,
            output_dir=tmp_path,
            node_id="tabular_out",
            columns=["id", "col1"],
            cv=False,
            splits=None,
        )
        loaded = pl.read_parquet(tmp_path / "tabular_out" / "test.parquet")
        assert loaded.columns == ["id", "col1"]
        assert len(loaded) == 10


class TestOutputWithCV:
    def test_fold_directories_created(
        self, resolver: OutputResolver, sample_df: pl.DataFrame, tmp_path: Path
    ) -> None:
        """cv=true の場合、fold_0 / fold_1 ディレクトリが生成されること。"""
        # n_splits=2 のシンプルな splits を作成
        splits = [
            (list(range(5)), list(range(5, 10))),  # fold 0: train=0-4, test=5-9
            (list(range(5, 10)), list(range(0, 5))),  # fold 1: train=5-9, test=0-4
        ]
        resolver.output(
            df=sample_df,
            output_dir=tmp_path,
            node_id="tabular_out",
            columns=["id", "col1", "label"],
            cv=True,
            splits=splits,
        )
        assert (tmp_path / "tabular_out" / "fold_0" / "train.parquet").exists()
        assert (tmp_path / "tabular_out" / "fold_0" / "test.parquet").exists()
        assert (tmp_path / "tabular_out" / "fold_1" / "train.parquet").exists()
        assert (tmp_path / "tabular_out" / "fold_1" / "test.parquet").exists()

    def test_fold_row_counts(
        self, resolver: OutputResolver, sample_df: pl.DataFrame, tmp_path: Path
    ) -> None:
        """fold_0/train.parquet が train_indices の行数と一致すること。"""
        splits = [
            (list(range(7)), list(range(7, 10))),  # fold 0: train=7行, test=3行
        ]
        resolver.output(
            df=sample_df,
            output_dir=tmp_path,
            node_id="tabular_out",
            columns=["id", "col1", "label"],
            cv=True,
            splits=splits,
        )
        train_df = pl.read_parquet(tmp_path / "tabular_out" / "fold_0" / "train.parquet")
        test_df = pl.read_parquet(tmp_path / "tabular_out" / "fold_0" / "test.parquet")
        assert len(train_df) == 7
        assert len(test_df) == 3
