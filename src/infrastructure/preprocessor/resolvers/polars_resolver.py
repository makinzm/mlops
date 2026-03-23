"""
Polars ベースの変換 Resolver。

対応メソッド:
- select_columns : 指定カラムのみ残す
- arithmetic     : 四則演算 + log1p（add/subtract/multiply/divide/log1p）
- exp_weight     : 時系列指数重みカラム（__weight__ 等）を追加
- join           : 複数 DataFrame の left/right/inner join
"""

import math

import polars as pl


class PolarsResolver:
    """Polars を使ったタブラーデータ変換 Resolver。"""

    def supported_methods(self) -> set[str]:
        return {"select_columns", "arithmetic", "exp_weight", "join"}

    def execute(self, df: pl.DataFrame, method: str, **kwargs: object) -> pl.DataFrame:
        """メソッド名に応じて変換を実行する。"""
        if method == "select_columns":
            columns = kwargs.get("columns")
            if not isinstance(columns, list):
                raise ValueError("select_columns requires 'columns' as list[str]")
            return self.select_columns(df, columns=columns)  # ty:ignore[invalid-argument-type]

        if method == "arithmetic":
            operation = str(kwargs.get("operation", ""))
            col_a = str(kwargs.get("col_a", ""))
            col_b = kwargs.get("col_b")
            output_col = str(kwargs.get("output_col", ""))
            return self.arithmetic(
                df,
                operation=operation,
                col_a=col_a,
                col_b=str(col_b) if col_b is not None else None,
                output_col=output_col,
            )

        if method == "exp_weight":
            time_col = str(kwargs.get("time_col", ""))
            decay = float(kwargs.get("decay", 0.95))  # ty:ignore[invalid-argument-type]
            weight_col = str(kwargs.get("weight_col", "__weight__"))
            return self.exp_weight(df, time_col=time_col, decay=decay, weight_col=weight_col)

        if method == "join":
            dfs = kwargs.get("dfs")
            if not isinstance(dfs, list):
                raise ValueError("join requires 'dfs' as list[pl.DataFrame]")
            on = str(kwargs.get("on", ""))
            how = str(kwargs.get("how", "left"))
            return self.join(dfs, on=on, how=how)  # ty:ignore[invalid-argument-type]

        raise ValueError(f"Unknown method: {method!r}")

    # -------------------------------------------------------------------
    # Individual method implementations
    # -------------------------------------------------------------------

    def select_columns(self, df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
        """指定カラムのみを残す。"""
        return df.select(columns)

    def arithmetic(
        self,
        df: pl.DataFrame,
        operation: str,
        col_a: str,
        output_col: str,
        col_b: str | None = None,
    ) -> pl.DataFrame:
        """四則演算または log1p を行い、output_col カラムを追加する。

        operation:
            "add"      : col_a + col_b
            "subtract" : col_a - col_b
            "multiply" : col_a * col_b
            "divide"   : col_a / col_b
            "log1p"    : log(1 + col_a)（col_b 不要）
        """
        if operation == "add":
            if col_b is None:
                raise ValueError("'add' requires col_b")
            expr = pl.col(col_a) + pl.col(col_b)
        elif operation == "subtract":
            if col_b is None:
                raise ValueError("'subtract' requires col_b")
            expr = pl.col(col_a) - pl.col(col_b)
        elif operation == "multiply":
            if col_b is None:
                raise ValueError("'multiply' requires col_b")
            expr = pl.col(col_a) * pl.col(col_b)
        elif operation == "divide":
            if col_b is None:
                raise ValueError("'divide' requires col_b")
            expr = pl.col(col_a) / pl.col(col_b)
        elif operation == "log1p":
            expr = (pl.col(col_a) + pl.lit(1.0)).log(math.e)
        else:
            raise ValueError(f"Unknown operation: {operation!r}")

        return df.with_columns(expr.alias(output_col))

    def exp_weight(
        self,
        df: pl.DataFrame,
        time_col: str,
        decay: float,
        weight_col: str = "__weight__",
    ) -> pl.DataFrame:
        """時系列指数重みカラムを追加する。

        行の時系列順位（0始まり昇順）を rank とし、
        weight = decay^(max_rank - rank) で計算する。
        最新（最大 rank）の重みが 1.0 になる。
        """
        n = len(df)
        # 時系列カラムで昇順ランクを付ける
        ranked = df.with_row_index("__row_idx__")
        sorted_idx = ranked.sort(time_col).with_columns(pl.arange(0, n).alias("__rank__"))
        # __row_idx__ でもとの順序に戻す
        with_rank = ranked.join(
            sorted_idx.select(["__row_idx__", "__rank__"]), on="__row_idx__"
        ).drop("__row_idx__")
        max_rank = n - 1
        with_weight = with_rank.with_columns(
            (pl.lit(decay) ** (pl.lit(max_rank) - pl.col("__rank__"))).alias(weight_col)
        ).drop("__rank__")
        return with_weight

    def join(
        self,
        dfs: list[pl.DataFrame],
        on: str,
        how: str = "left",
    ) -> pl.DataFrame:
        """複数 DataFrame を順番に join する。

        dfs[0] を左テーブルとし、dfs[1:] を順次 join する。
        """
        if len(dfs) < 2:
            raise ValueError("join requires at least 2 DataFrames")
        result = dfs[0]
        for right in dfs[1:]:
            result = result.join(right, on=on, how=how)  # ty:ignore[invalid-argument-type]
        return result
