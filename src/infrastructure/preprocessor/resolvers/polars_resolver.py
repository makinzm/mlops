"""
Polars ベースの変換 Resolver。

対応メソッド:
- select_columns            : 指定カラムのみ残す
- arithmetic                : 四則演算 + log1p（add/subtract/multiply/divide/log1p）
- exp_weight                : 時系列指数重みカラム（__weight__ 等）を追加
- join                      : 複数 DataFrame の left/right/inner join
- bayesian_target_encode    : Bayesian Target Encoding（Beta-Binomial / Normal-Gamma）
- time_series_target_encode : 時系列 Expanding Window Target Encoding
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import numpy as np
import polars as pl
from sklearn.model_selection import KFold

# カラム指定の型: 単独カラム名 or 複合キー（複数カラム名のリスト）
ColumnSpec = str | list[str]


class BayesianStats(NamedTuple):
    """Bayesian Target Encoding の事後統計量。"""

    posterior_mean: float
    posterior_var: float
    alpha_post: float
    beta_post: float
    n_samples: int


# カラム名（元名 or 複合結合名） → {カテゴリ値: BayesianStats, "__prior__": BayesianStats}
BayesianTargetEncoder = dict[str, dict[str, BayesianStats]]


class PolarsResolver:
    """Polars を使ったタブラーデータ変換 Resolver。"""

    def supported_methods(self) -> set[str]:
        return {
            "select_columns",
            "arithmetic",
            "exp_weight",
            "join",
            "bayesian_target_encode",
            "time_series_target_encode",
        }

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

    # -------------------------------------------------------------------
    # Bayesian Target Encoding
    # -------------------------------------------------------------------

    @staticmethod
    def _resolve_column_spec(
        spec: ColumnSpec,
    ) -> tuple[str, list[str]]:
        """ColumnSpec を (出力名, 構成カラムリスト) に解決する。

        単独キー "Sex" → ("Sex", ["Sex"])
        複合キー ["Sex", "Embarked"] → ("Sex_Embarked", ["Sex", "Embarked"])
        """
        if isinstance(spec, list):
            return ("_".join(spec), spec)
        return (spec, [spec])

    @staticmethod
    def _build_group_column(
        df: pl.DataFrame, source_columns: list[str]
    ) -> tuple[pl.DataFrame, str]:
        """複合キー用の一時グループカラムを作成する。

        単独カラムの場合はそのまま返す。
        複合キーの場合は "__te_group__" カラムを追加して返す。
        """
        if len(source_columns) == 1:
            return df, source_columns[0]
        group_col = "__te_group__"
        concat_expr = pl.concat_str(
            [pl.col(c).cast(pl.Utf8) for c in source_columns], separator="_"
        )
        return df.with_columns(concat_expr.alias(group_col)), group_col

    @staticmethod
    def _compute_binary_stats(
        target_values: list[float],
        global_mean: float,
        prior_weight: float,
    ) -> BayesianStats:
        """二値分類（Beta-Binomial）の事後統計量を計算する。

        事前分布: Beta(α₀, β₀)
          α₀ = global_mean × prior_weight
          β₀ = (1 - global_mean) × prior_weight
        事後分布: Beta(α₀ + successes, β₀ + failures)
        事後平均: α_post / (α_post + β_post)
        事後分散: α_post × β_post / ((α_post + β_post)² × (α_post + β_post + 1))

        時間計算量: O(N) where N = len(target_values)
        空間計算量: O(1)
        """
        alpha_0 = global_mean * prior_weight
        beta_0 = (1.0 - global_mean) * prior_weight
        successes = sum(target_values)
        n = len(target_values)
        failures = n - successes

        alpha_post = alpha_0 + successes
        beta_post = beta_0 + failures
        total = alpha_post + beta_post

        posterior_mean = alpha_post / total if total > 0 else global_mean
        posterior_var = (
            (alpha_post * beta_post) / (total * total * (total + 1.0)) if total > 0 else 0.0
        )
        return BayesianStats(
            posterior_mean=posterior_mean,
            posterior_var=posterior_var,
            alpha_post=alpha_post,
            beta_post=beta_post,
            n_samples=n,
        )

    @staticmethod
    def _compute_continuous_stats(
        target_values: list[float],
        global_mean: float,
        prior_weight: float,
    ) -> BayesianStats:
        """連続値（Normal-Gamma）の事後統計量を計算する。

        事前分布: Normal-Gamma(μ₀, κ₀, α₀, β₀)
          μ₀ = global_mean, κ₀ = prior_weight, α₀ = 1.0, β₀ = 1.0
        事後平均: (κ₀ × μ₀ + N × mean_i) / (κ₀ + N)
        事後分散: β_post / (α_post × κ_post)

        時間計算量: O(N) where N = len(target_values)
        空間計算量: O(1)
        """
        mu_0 = global_mean
        kappa_0 = prior_weight
        alpha_0 = 1.0
        beta_0 = 1.0
        n = len(target_values)

        if n == 0:
            return BayesianStats(
                posterior_mean=global_mean,
                posterior_var=beta_0 / (alpha_0 * kappa_0) if kappa_0 > 0 else 0.0,
                alpha_post=alpha_0,
                beta_post=beta_0,
                n_samples=0,
            )

        sample_mean = sum(target_values) / n
        kappa_post = kappa_0 + n
        alpha_post = alpha_0 + n / 2.0
        posterior_mean = (kappa_0 * mu_0 + n * sample_mean) / kappa_post

        # β_post = β₀ + 0.5 * Σ(x_i - x̄)² + κ₀*n*(x̄ - μ₀)² / (2*κ_post)
        ss = sum((x - sample_mean) ** 2 for x in target_values)
        beta_post = beta_0 + 0.5 * ss + kappa_0 * n * (sample_mean - mu_0) ** 2 / (2.0 * kappa_post)
        posterior_var = beta_post / (alpha_post * kappa_post) if kappa_post > 0 else 0.0

        return BayesianStats(
            posterior_mean=posterior_mean,
            posterior_var=posterior_var,
            alpha_post=alpha_post,
            beta_post=beta_post,
            n_samples=n,
        )

    def _compute_stats(
        self,
        target_values: list[float],
        global_mean: float,
        prior_weight: float,
        target_type: Literal["binary", "continuous"],
    ) -> BayesianStats:
        """target_type に応じた事後統計量を計算する。"""
        if target_type == "binary":
            return self._compute_binary_stats(target_values, global_mean, prior_weight)
        return self._compute_continuous_stats(target_values, global_mean, prior_weight)

    def _fit_bayesian_encoder(
        self,
        group_values: list[str],
        target_values: np.ndarray,  # ty:ignore[unresolved-reference]
        target_type: Literal["binary", "continuous"],
        prior_weight: float,
        min_samples_leaf: int,
    ) -> tuple[dict[str, BayesianStats], BayesianStats]:
        """全データで Bayesian encoder を fit する。

        時間計算量: O(N × C) where N = データ数, C = カテゴリ数
        空間計算量: O(C) カテゴリごとの統計量を保持

        Returns:
            (category_stats, prior_stats)
        """
        global_mean = float(np.mean(target_values))
        prior_stats = self._compute_stats(
            target_values.tolist(), global_mean, prior_weight, target_type
        )

        # カテゴリごとに target 値を集める
        cat_targets: dict[str, list[float]] = {}
        for cat, t in zip(group_values, target_values):
            cat_targets.setdefault(cat, []).append(float(t))

        category_stats: dict[str, BayesianStats] = {}
        for cat, targets in cat_targets.items():
            if len(targets) < min_samples_leaf:
                category_stats[cat] = BayesianStats(
                    posterior_mean=prior_stats.posterior_mean,
                    posterior_var=prior_stats.posterior_var,
                    alpha_post=prior_stats.alpha_post,
                    beta_post=prior_stats.beta_post,
                    n_samples=len(targets),
                )
            else:
                category_stats[cat] = self._compute_stats(
                    targets, global_mean, prior_weight, target_type
                )

        return category_stats, prior_stats

    def bayesian_target_encode(
        self,
        df: pl.DataFrame,
        columns: list[ColumnSpec],
        target_col: str,
        target_type: Literal["binary", "continuous"] = "binary",
        n_splits: int = 5,
        seed: int = 42,
        prior_weight: float = 1.0,
        min_samples_leaf: int = 1,
        output_variance: bool = True,
        suffix: str = "",
        prefix: str = "",
    ) -> tuple[pl.DataFrame, BayesianTargetEncoder]:
        """Bayesian Target Encoding（OOF CV）を行う。

        二値分類では Beta-Binomial、連続値では Normal-Gamma の事後平均を使う。
        OOF CV により val fold のデータを encoder の fit に使わずリークを防ぐ。
        複合キーグルーピング対応: columns に list[str] を含めることで交互作用を捉える。

        時間計算量: O(K × N × C) where K = n_splits, N = データ数, C = カテゴリ数
        空間計算量: O(N + C) OOF 配列 + カテゴリ統計量

        Args:
            df: Train DataFrame（target_col を含む）
            columns: エンコード対象の ColumnSpec リスト
                - "Sex" → 単独カラムで TE
                - ["Sex", "Embarked"] → 複合キーで TE（出力: Sex_Embarked{suffix}）
            target_col: ターゲットカラム名
            target_type: "binary"（Beta-Binomial）or "continuous"（Normal-Gamma）
            n_splits: KFold の分割数
            seed: KFold のシード
            prior_weight: 事前分布の重み（大 → global_mean 寄り）
            min_samples_leaf: この数未満のカテゴリは global_mean にフォールバック
            output_variance: True なら事後分散カラムも出力する
            suffix: 出力カラム名の接尾辞（デフォルト空文字 → "_te" が自動付与）
            prefix: 出力カラム名の接頭辞

        Returns:
            (oof_encoded_df, full_encoder)
        """
        effective_suffix = suffix if suffix else "_te"
        target = df[target_col].cast(pl.Float64).to_numpy()
        n = len(df)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        result = df.clone()
        full_encoder: BayesianTargetEncoder = {}

        for spec in columns:
            col_name, source_cols = self._resolve_column_spec(spec)
            output_col = f"{prefix}{col_name}{effective_suffix}"

            # 複合キーのグループカラムを作成
            working_df, group_col = self._build_group_column(df, source_cols)
            group_values = [str(v) for v in working_df[group_col].to_list()]

            oof_encoded = np.full(n, np.nan, dtype=np.float64)
            oof_variance = np.full(n, np.nan, dtype=np.float64) if output_variance else None

            # OOF エンコード
            for train_idx, val_idx in kf.split(np.arange(n)):
                fold_group = [group_values[i] for i in train_idx]
                fold_target = target[train_idx]
                fold_global_mean = float(np.mean(fold_target))

                # fold の train でカテゴリ統計量を計算
                cat_targets: dict[str, list[float]] = {}
                for cat, t in zip(fold_group, fold_target):
                    cat_targets.setdefault(cat, []).append(float(t))

                # val を fold の統計量でエンコード
                for i in val_idx:
                    cat = group_values[i]
                    if cat in cat_targets and len(cat_targets[cat]) >= min_samples_leaf:
                        stats = self._compute_stats(
                            cat_targets[cat], fold_global_mean, prior_weight, target_type
                        )
                    else:
                        stats = self._compute_stats(
                            fold_target.tolist(),
                            fold_global_mean,
                            prior_weight,
                            target_type,
                        )
                    oof_encoded[i] = stats.posterior_mean
                    if oof_variance is not None:
                        oof_variance[i] = stats.posterior_var

            result = result.with_columns(pl.Series(output_col, oof_encoded))
            if output_variance and oof_variance is not None:
                result = result.with_columns(pl.Series(f"{output_col}_var", oof_variance))

            # full_encoder: 全 train で fit
            category_stats, prior_stats = self._fit_bayesian_encoder(
                group_values, target, target_type, prior_weight, min_samples_leaf
            )
            col_encoder: dict[str, BayesianStats] = {"__prior__": prior_stats}
            col_encoder.update(category_stats)
            full_encoder[col_name] = col_encoder

        return result, full_encoder

    def transform_bayesian_target_encode(
        self,
        df: pl.DataFrame,
        encoder: BayesianTargetEncoder,
        columns: list[ColumnSpec],
        suffix: str = "",
        prefix: str = "",
    ) -> pl.DataFrame:
        """full encoder を使って df（Test データ）を Bayesian Target Encoding する。

        未知カテゴリは encoder[col]["__prior__"].posterior_mean でフォールバック。

        時間計算量: O(N × C_cols) where N = 行数, C_cols = 対象カラム数
        空間計算量: O(N) エンコード結果配列

        Args:
            df: Test DataFrame
            encoder: bayesian_target_encode() が返した full_encoder
            columns: エンコード対象 ColumnSpec リスト
            suffix: 出力カラム名の接尾辞（デフォルト空文字 → "_te" が自動付与）
            prefix: 出力カラム名の接頭辞

        Returns:
            エンコード済み DataFrame
        """
        effective_suffix = suffix if suffix else "_te"
        result = df.clone()

        for spec in columns:
            col_name, source_cols = self._resolve_column_spec(spec)
            output_col = f"{prefix}{col_name}{effective_suffix}"

            working_df, group_col = self._build_group_column(df, source_cols)
            group_values = [str(v) for v in working_df[group_col].to_list()]

            col_encoder = encoder[col_name]
            prior = col_encoder["__prior__"]
            encoded = [col_encoder.get(cat, prior).posterior_mean for cat in group_values]
            result = result.with_columns(pl.Series(output_col, encoded, dtype=pl.Float64))

        return result

    # -------------------------------------------------------------------
    # Time Series Target Encoding (Expanding Window)
    # -------------------------------------------------------------------

    def time_series_target_encode(
        self,
        df: pl.DataFrame,
        columns: list[ColumnSpec],
        target_col: str,
        time_col: str,
        target_type: Literal["binary", "continuous"] = "binary",
        prior_weight: float = 1.0,
        min_samples: int = 1,
        suffix: str = "",
        prefix: str = "",
    ) -> tuple[pl.DataFrame, BayesianTargetEncoder]:
        """時系列 Expanding Window Target Encoding を行う。

        各行で自分より過去のデータのみから Bayesian 統計量を計算し、
        未来の情報を使わないことでデータリークを防ぐ。
        履歴がない行は事前分布の平均（global_mean）で埋める。

        時間計算量: O(N² × C) where N = データ数, C = カテゴリ数
          各行で過去の全データを集計するため N² になる。
        空間計算量: O(N + C) 結果配列 + カテゴリ統計量

        Args:
            df: Train DataFrame（target_col, time_col を含む）
            columns: エンコード対象の ColumnSpec リスト
            target_col: ターゲットカラム名
            time_col: 時系列カラム名（ソート用）
            target_type: "binary"（Beta-Binomial）or "continuous"（Normal-Gamma）
            prior_weight: 事前分布の重み
            min_samples: この数未満の履歴は事前分布の平均でフォールバック
            suffix: 出力カラム名の接尾辞（デフォルト空文字 → "_te" が自動付与）
            prefix: 出力カラム名の接頭辞

        Returns:
            (encoded_df, full_encoder)
            full_encoder は全 train で fit した encoder（transform 用）
        """
        effective_suffix = suffix if suffix else "_te"
        n = len(df)
        target = df[target_col].cast(pl.Float64).to_numpy()
        global_mean = float(np.mean(target))

        # 元の行順序を保持するために row_index を付与
        indexed_df = df.with_row_index("__orig_idx__")
        sorted_df = indexed_df.sort(time_col)
        sorted_indices = sorted_df["__orig_idx__"].to_list()

        result = df.clone()
        full_encoder: BayesianTargetEncoder = {}

        for spec in columns:
            col_name, source_cols = self._resolve_column_spec(spec)
            output_col = f"{prefix}{col_name}{effective_suffix}"

            # 複合キーのグループ値を元の順序で取得
            working_df, group_col = self._build_group_column(df, source_cols)
            all_group_values = [str(v) for v in working_df[group_col].to_list()]

            encoded_values = np.full(n, np.nan, dtype=np.float64)

            # 時系列順で expanding window
            for pos in range(n):
                orig_idx = sorted_indices[pos]
                current_cat = all_group_values[orig_idx]

                # 過去のデータ（pos 未満）から統計量を集める
                past_targets_for_cat: list[float] = []
                for past_pos in range(pos):
                    past_orig_idx = sorted_indices[past_pos]
                    if all_group_values[past_orig_idx] == current_cat:
                        past_targets_for_cat.append(float(target[past_orig_idx]))

                if len(past_targets_for_cat) < min_samples:
                    encoded_values[orig_idx] = global_mean
                else:
                    stats = self._compute_stats(
                        past_targets_for_cat, global_mean, prior_weight, target_type
                    )
                    encoded_values[orig_idx] = stats.posterior_mean

            result = result.with_columns(pl.Series(output_col, encoded_values))

            # full_encoder: 全 train で fit（transform 用）
            category_stats, prior_stats = self._fit_bayesian_encoder(
                all_group_values, target, target_type, prior_weight, min_samples
            )
            col_encoder: dict[str, BayesianStats] = {"__prior__": prior_stats}
            col_encoder.update(category_stats)
            full_encoder[col_name] = col_encoder

        return result, full_encoder
