"""
OutputResolver — パイプラインの最終出力ステップ。

対応メソッド:
- output : 指定カラムを Parquet で書き出す。
           cv=false → {output_dir}/{node_id}/test.parquet（サブディレクトリ）
           cv=true  → {output_dir}/{node_id}/fold_N/train.parquet + test.parquet

なぜ cv=false もサブディレクトリ形式にするか:
  InferenceUseCase が {preprocess_output_dir}/test_out/test.parquet を期待するため。

OutputResolver だけは DataFrame を変換せず、ファイルに書き出す副作用を持つ。
戻り値は元の DataFrame をそのまま返す（後続ステップから参照可能にするため）。
"""

from pathlib import Path

import polars as pl


class OutputResolver:
    """Parquet 書き出し Resolver。"""

    def supported_methods(self) -> set[str]:
        return {"output"}

    def execute(self, df: pl.DataFrame, method: str, **kwargs: object) -> pl.DataFrame:
        """output メソッドを実行する。"""
        if method == "output":
            output_dir = kwargs.get("output_dir")
            if not isinstance(output_dir, Path):
                raise ValueError("output requires 'output_dir' as Path")
            node_id = str(kwargs.get("node_id", "output"))
            columns = kwargs.get("columns")
            if not isinstance(columns, list):
                raise ValueError("output requires 'columns' as list[str]")
            cv = bool(kwargs.get("cv", False))
            splits = kwargs.get("splits")
            self.output(
                df=df,
                output_dir=output_dir,
                node_id=node_id,
                columns=columns,
                cv=cv,
                splits=splits,  # type: ignore[arg-type]
            )
            return df
        raise ValueError(f"Unknown method: {method!r}")

    def output(
        self,
        df: pl.DataFrame,
        output_dir: Path,
        node_id: str,
        columns: list[str],
        cv: bool,
        splits: list[tuple[list[int], list[int]]] | None,
    ) -> None:
        """指定カラムを Parquet で書き出す。

        Args:
            df:         書き出す DataFrame
            output_dir: 出力ルートディレクトリ
            node_id:    出力ノード id（ファイル名・ディレクトリ名に使用）
            columns:    書き出すカラム名リスト
            cv:         True の場合 fold ディレクトリに分割して書き出す
            splits:     [(train_indices, test_indices), ...] の fold リスト
                        cv=True の場合は必須
        """
        subset = df.select(columns)

        if not cv:
            # サブディレクトリ形式: {output_dir}/{node_id}/test.parquet
            # InferenceUseCase が {preprocess_output_dir}/{node_id}/test.parquet を期待するため
            out_dir = output_dir / node_id
            out_dir.mkdir(parents=True, exist_ok=True)
            subset.write_parquet(out_dir / "test.parquet")
        else:
            if splits is None:
                raise ValueError("splits must be provided when cv=True")
            for fold_idx, (train_indices, test_indices) in enumerate(splits):
                fold_dir = output_dir / node_id / f"fold_{fold_idx}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                subset[train_indices].write_parquet(fold_dir / "train.parquet")
                subset[test_indices].write_parquet(fold_dir / "test.parquet")
