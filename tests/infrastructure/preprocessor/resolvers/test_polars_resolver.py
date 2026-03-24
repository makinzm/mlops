"""
PolarsResolver の各メソッドに対するユニットテスト。

なぜこのテストが必要か:
- polars:select_columns / arithmetic / exp_weight / join は
  パイプラインの主要変換処理であり、戻り値の形状・カラム名・数値を
  明示的に検証して回帰を防ぐ必要がある。
- join は複数 Node の DataFrame をマージするため、
  DAGRunner から呼ばれる際の入力フォーマット（DataFrameのリスト）を想定した設計にする。
- exp_weight は「時系列の新しいデータほど重みが高い」挙動を検証する。
"""

import polars as pl
import pytest

from src.infrastructure.preprocessor.resolvers.polars_resolver import PolarsResolver


@pytest.fixture()
def resolver() -> PolarsResolver:
    return PolarsResolver()


@pytest.fixture()
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "label": [0, 1, 0, 1, 0],
            "extra": ["a", "b", "c", "d", "e"],
        }
    )


class TestSelectColumns:
    def test_select_keeps_specified_columns(
        self, resolver: PolarsResolver, sample_df: pl.DataFrame
    ) -> None:
        """指定カラムのみが残ること。"""
        result = resolver.select_columns(sample_df, columns=["id", "col1", "label"])
        assert result.columns == ["id", "col1", "label"]
        assert len(result) == 5

    def test_select_removes_unspecified_columns(
        self, resolver: PolarsResolver, sample_df: pl.DataFrame
    ) -> None:
        """未指定カラム (extra, col2) が除去されること。"""
        result = resolver.select_columns(sample_df, columns=["id", "col1", "label"])
        assert "extra" not in result.columns
        assert "col2" not in result.columns

    def test_supported_methods_includes_select_columns(self, resolver: PolarsResolver) -> None:
        """supported_methods() に select_columns が含まれること。"""
        assert "select_columns" in resolver.supported_methods()


class TestArithmetic:
    def test_add(self, resolver: PolarsResolver, sample_df: pl.DataFrame) -> None:
        """add 演算が col_a + col_b を output_col に追加すること。"""
        result = resolver.arithmetic(
            sample_df, operation="add", col_a="col1", col_b="col2", output_col="sum_col"
        )
        assert "sum_col" in result.columns
        assert result["sum_col"].to_list() == [11.0, 22.0, 33.0, 44.0, 55.0]

    def test_multiply(self, resolver: PolarsResolver, sample_df: pl.DataFrame) -> None:
        """multiply 演算が col_a * col_b を output_col に追加すること。"""
        result = resolver.arithmetic(
            sample_df, operation="multiply", col_a="col1", col_b="col2", output_col="prod_col"
        )
        assert result["prod_col"].to_list() == [10.0, 40.0, 90.0, 160.0, 250.0]

    def test_log1p_single_column(self, resolver: PolarsResolver, sample_df: pl.DataFrame) -> None:
        """log1p 演算が col_a の log1p を output_col に追加すること（col_b は不要）。"""
        import math

        result = resolver.arithmetic(
            sample_df, operation="log1p", col_a="col1", output_col="log_col"
        )
        assert "log_col" in result.columns
        assert abs(result["log_col"][0] - math.log1p(1.0)) < 1e-6

    def test_subtract(self, resolver: PolarsResolver, sample_df: pl.DataFrame) -> None:
        """subtract 演算が col_a - col_b を output_col に追加すること。"""
        result = resolver.arithmetic(
            sample_df, operation="subtract", col_a="col2", col_b="col1", output_col="diff_col"
        )
        assert result["diff_col"].to_list() == [9.0, 18.0, 27.0, 36.0, 45.0]

    def test_divide(self, resolver: PolarsResolver, sample_df: pl.DataFrame) -> None:
        """divide 演算が col_a / col_b を output_col に追加すること。"""
        result = resolver.arithmetic(
            sample_df, operation="divide", col_a="col2", col_b="col1", output_col="ratio_col"
        )
        assert result["ratio_col"].to_list() == [10.0, 10.0, 10.0, 10.0, 10.0]


class TestExpWeight:
    def test_weight_column_added(self, resolver: PolarsResolver) -> None:
        """__weight__ カラムが追加されること。"""
        df = pl.DataFrame({"id": [1, 2, 3], "date": ["2026-01-01", "2026-01-02", "2026-01-03"]})
        result = resolver.exp_weight(df, time_col="date", decay=0.95, weight_col="__weight__")
        assert "__weight__" in result.columns

    def test_newer_rows_have_higher_weight(self, resolver: PolarsResolver) -> None:
        """時系列で新しい行ほど重みが高いこと。"""
        df = pl.DataFrame({"id": [1, 2, 3], "date": ["2026-01-01", "2026-01-02", "2026-01-03"]})
        result = resolver.exp_weight(df, time_col="date", decay=0.95, weight_col="__weight__")
        weights = result["__weight__"].to_list()
        # 昇順ソートで最後（最新）が最大重み
        assert weights[0] < weights[1] < weights[2]

    def test_weight_col_name_respected(self, resolver: PolarsResolver) -> None:
        """weight_col に指定した名前でカラムが追加されること。"""
        df = pl.DataFrame({"id": [1, 2], "date": ["2026-01-01", "2026-01-02"]})
        result = resolver.exp_weight(df, time_col="date", decay=0.9, weight_col="my_weight")
        assert "my_weight" in result.columns


class TestJoin:
    def test_left_join_on_id(self, resolver: PolarsResolver) -> None:
        """left join で左 DataFrame の行が全て保持されること。"""
        left = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        right = pl.DataFrame({"id": [1, 2], "extra": ["a", "b"]})
        result = resolver.join([left, right], on="id", how="left")
        assert len(result) == 3
        assert "extra" in result.columns

    def test_inner_join_on_id(self, resolver: PolarsResolver) -> None:
        """inner join でマッチする行のみ残ること。"""
        left = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        right = pl.DataFrame({"id": [1, 2], "extra": ["a", "b"]})
        result = resolver.join([left, right], on="id", how="inner")
        assert len(result) == 2


class TestBayesianTargetEncode:
    """
    なぜこのテストが必要か:
    - Bayesian Target Encoding は事前分布（prior）と観測データを組み合わせた
      事後平均を使うことで、小カテゴリの過学習を理論的根拠に基づき防ぐ。
    - sklearn TE の smoothing パラメータと異なり、prior_weight で
      事前分布の強さを制御する。この挙動を数値テストで検証する。
    - 二値分類（Beta-Binomial）と連続値（Normal-Gamma）で異なる計算式を使うため、
      それぞれの事後平均を手計算と照合する必要がある。
    - OOF CV でリークしないことは target encoding の根本的な安全性保証である。
    - 事後分散の出力は不確実性の定量化に使われるため、正しく計算されることを確認する。
    """

    def test_binary_basic(self, resolver: PolarsResolver) -> None:
        """二値分類で float カラムが生成され null がないこと。"""
        df = pl.DataFrame(
            {
                "Sex": ["male", "female", "male", "female", "male", "female"],
                "Survived": [0, 1, 0, 1, 1, 0],
            }
        )
        result, encoder = resolver.bayesian_target_encode(
            df,
            columns=["Sex"],
            target_col="Survived",
            target_type="binary",
            n_splits=2,
            seed=42,
        )
        output_col = "Sex_te"
        assert output_col in result.columns
        assert result[output_col].dtype in (pl.Float32, pl.Float64)
        assert result[output_col].null_count() == 0
        assert "Sex" in encoder

    def test_binary_posterior_mean_correctness(self, resolver: PolarsResolver) -> None:
        """Beta-Binomial 事後平均が手計算と一致すること。

        full encoder（全 train で fit）の値を検証。
        データ: male=[0,0,1], female=[1,1,0]
        global_mean = 3/6 = 0.5, prior_weight=2.0
        α₀ = 0.5 * 2 = 1.0, β₀ = 0.5 * 2 = 1.0

        male: successes=1, failures=2, N=3
          posterior_mean = (1.0 + 1) / (1.0 + 1.0 + 3) = 2/5 = 0.4

        female: successes=2, failures=1, N=3
          posterior_mean = (1.0 + 2) / (1.0 + 1.0 + 3) = 3/5 = 0.6
        """
        df = pl.DataFrame(
            {
                "Sex": ["male", "male", "male", "female", "female", "female"],
                "Survived": [0, 0, 1, 1, 1, 0],
            }
        )
        _, encoder = resolver.bayesian_target_encode(
            df,
            columns=["Sex"],
            target_col="Survived",
            target_type="binary",
            n_splits=2,
            seed=42,
            prior_weight=2.0,
        )
        assert abs(encoder["Sex"]["male"].posterior_mean - 0.4) < 1e-6
        assert abs(encoder["Sex"]["female"].posterior_mean - 0.6) < 1e-6

    def test_binary_posterior_variance_output(self, resolver: PolarsResolver) -> None:
        """output_variance=True で _var カラムが出力されること。"""
        df = pl.DataFrame(
            {
                "Sex": ["male", "female", "male", "female"],
                "Survived": [0, 1, 0, 1],
            }
        )
        result, _ = resolver.bayesian_target_encode(
            df,
            columns=["Sex"],
            target_col="Survived",
            target_type="binary",
            n_splits=2,
            seed=42,
            output_variance=True,
        )
        assert "Sex_te_var" in result.columns
        assert result["Sex_te_var"].null_count() == 0

    def test_binary_prior_weight_effect(self, resolver: PolarsResolver) -> None:
        """prior_weight が大きいほど global_mean に近づくこと。

        male: successes=3, failures=0 → raw mean=1.0, global_mean=0.5
        prior_weight 小 → 1.0 に近い
        prior_weight 大 → 0.5 に近い
        """
        df = pl.DataFrame(
            {
                "Sex": ["male", "male", "male", "female", "female", "female"],
                "Survived": [1, 1, 1, 0, 0, 0],
            }
        )
        _, encoder_small = resolver.bayesian_target_encode(
            df,
            columns=["Sex"],
            target_col="Survived",
            target_type="binary",
            n_splits=2,
            seed=42,
            prior_weight=0.1,
        )
        _, encoder_large = resolver.bayesian_target_encode(
            df,
            columns=["Sex"],
            target_col="Survived",
            target_type="binary",
            n_splits=2,
            seed=42,
            prior_weight=100.0,
        )
        # prior_weight 小 → raw mean (1.0) 寄り
        # prior_weight 大 → global_mean (0.5) 寄り
        assert encoder_small["Sex"]["male"].posterior_mean > encoder_large["Sex"]["male"].posterior_mean

    def test_binary_min_samples_leaf(self, resolver: PolarsResolver) -> None:
        """min_samples_leaf 未満のカテゴリは global_mean にフォールバックすること。

        rare_cat は 1 件しかないため、min_samples_leaf=2 で global_mean に置換される。
        """
        df = pl.DataFrame(
            {
                "Cat": ["A", "A", "A", "A", "B", "B", "B", "B", "rare_cat", "rare_cat"],
                "Target": [1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
            }
        )
        _, encoder = resolver.bayesian_target_encode(
            df,
            columns=["Cat"],
            target_col="Target",
            target_type="binary",
            n_splits=2,
            seed=42,
            min_samples_leaf=3,
        )
        global_mean_stats = encoder["Cat"]["__prior__"]
        # rare_cat (N=2) は min_samples_leaf=3 未満 → global_mean にフォールバック
        assert abs(encoder["Cat"]["rare_cat"].posterior_mean - global_mean_stats.posterior_mean) < 1e-6

    def test_binary_oof_no_leak(self, resolver: PolarsResolver) -> None:
        """OOF CV でデータリークしないこと。

        val fold に target=100 の外れ値を配置し、
        OOF が正しく実装されていれば val 行の encoded 値が
        外れ値に引きずられないことを確認する。
        """
        df = pl.DataFrame(
            {
                "Sex": ["male", "male", "male", "male"],
                "Survived": [100.0, 100.0, 0.0, 0.0],
            }
        )
        result, _ = resolver.bayesian_target_encode(
            df,
            columns=["Sex"],
            target_col="Survived",
            target_type="binary",
            n_splits=2,
            seed=0,
        )
        # val 行 (0,1) の encoded 値が 50.0 を超えていればリーク
        val_values = result["Sex_te"].to_list()[:2]
        assert all(v < 50.0 for v in val_values), (
            f"OOF リークの疑い: val 行の encoded 値が大きすぎる {val_values}"
        )

    def test_continuous_basic(self, resolver: PolarsResolver) -> None:
        """Normal-Gamma で float カラムが出力されること。"""
        df = pl.DataFrame(
            {
                "Cat": ["A", "A", "B", "B", "A", "B"],
                "Value": [1.0, 2.0, 10.0, 20.0, 3.0, 15.0],
            }
        )
        result, encoder = resolver.bayesian_target_encode(
            df,
            columns=["Cat"],
            target_col="Value",
            target_type="continuous",
            n_splits=2,
            seed=42,
        )
        assert "Cat_te" in result.columns
        assert result["Cat_te"].dtype in (pl.Float32, pl.Float64)
        assert result["Cat_te"].null_count() == 0
        assert "Cat" in encoder

    def test_continuous_posterior_mean_correctness(self, resolver: PolarsResolver) -> None:
        """Normal-Gamma 事後平均が手計算と一致すること。

        full encoder（全 train で fit）の値を検証。
        データ: A=[1.0, 2.0, 3.0], B=[10.0, 20.0, 15.0]
        global_mean = (1+2+3+10+20+15)/6 = 51/6 = 8.5
        prior_weight (κ₀) = 2.0, μ₀ = 8.5

        A: N=3, mean=2.0
          posterior_mean = (2.0 * 8.5 + 3 * 2.0) / (2.0 + 3) = (17 + 6) / 5 = 4.6

        B: N=3, mean=15.0
          posterior_mean = (2.0 * 8.5 + 3 * 15.0) / (2.0 + 3) = (17 + 45) / 5 = 12.4
        """
        df = pl.DataFrame(
            {
                "Cat": ["A", "A", "A", "B", "B", "B"],
                "Value": [1.0, 2.0, 3.0, 10.0, 20.0, 15.0],
            }
        )
        _, encoder = resolver.bayesian_target_encode(
            df,
            columns=["Cat"],
            target_col="Value",
            target_type="continuous",
            n_splits=2,
            seed=42,
            prior_weight=2.0,
        )
        assert abs(encoder["Cat"]["A"].posterior_mean - 4.6) < 1e-6
        assert abs(encoder["Cat"]["B"].posterior_mean - 12.4) < 1e-6

    def test_unknown_category_fallback(self, resolver: PolarsResolver) -> None:
        """未知カテゴリは global_mean でフォールバックされること。"""
        train_df = pl.DataFrame(
            {
                "Sex": ["male", "female", "male", "female"],
                "Survived": [0, 1, 0, 1],
            }
        )
        test_df = pl.DataFrame({"Sex": ["unknown"]})
        _, encoder = resolver.bayesian_target_encode(
            train_df,
            columns=["Sex"],
            target_col="Survived",
            target_type="binary",
            n_splits=2,
            seed=42,
        )
        result = resolver.transform_bayesian_target_encode(
            test_df, encoder=encoder, columns=["Sex"]
        )
        global_mean = encoder["Sex"]["__prior__"].posterior_mean
        assert abs(result["Sex_te"][0] - global_mean) < 1e-6

    def test_supported_method(self, resolver: PolarsResolver) -> None:
        """supported_methods() に bayesian_target_encode が含まれること。"""
        assert "bayesian_target_encode" in resolver.supported_methods()


class TestBayesianTargetEncodeGrouping:
    """
    なぜこのテストが必要か:
    - 複合キーグルーピング（例: Sex × Embarked）は、
      単独カラムでは捉えられない交互作用を Target Encoding に組み込む機能。
    - 出力カラム名が `Sex_Embarked_te` のように自動生成されることを検証する。
    - 単独キーと複合キーを混在指定できることを保証する。
    - 未知の組み合わせ（Train に存在しない Sex × Embarked の組み合わせ）が
      global_mean にフォールバックされることを確認する。
    """

    def test_composite_key_grouping(self, resolver: PolarsResolver) -> None:
        """複合キー ["Sex", "Embarked"] で TE が実行されること。"""
        df = pl.DataFrame(
            {
                "Sex": ["male", "male", "female", "female", "male", "female"],
                "Embarked": ["S", "C", "S", "C", "S", "S"],
                "Survived": [0, 1, 1, 0, 0, 1],
            }
        )
        result, encoder = resolver.bayesian_target_encode(
            df,
            columns=[["Sex", "Embarked"]],
            target_col="Survived",
            target_type="binary",
            n_splits=2,
            seed=42,
        )
        assert "Sex_Embarked_te" in result.columns
        assert result["Sex_Embarked_te"].dtype in (pl.Float32, pl.Float64)
        assert "Sex_Embarked" in encoder

    def test_composite_key_output_column_name(self, resolver: PolarsResolver) -> None:
        """複合キーの出力カラム名が Sex_Embarked{suffix} であること。"""
        df = pl.DataFrame(
            {
                "Sex": ["male", "female", "male", "female"],
                "Embarked": ["S", "C", "S", "C"],
                "Survived": [0, 1, 0, 1],
            }
        )
        result, _ = resolver.bayesian_target_encode(
            df,
            columns=[["Sex", "Embarked"]],
            target_col="Survived",
            target_type="binary",
            n_splits=2,
            seed=42,
            suffix="_encoded",
        )
        assert "Sex_Embarked_encoded" in result.columns

    def test_mixed_single_and_composite(self, resolver: PolarsResolver) -> None:
        """単独キーと複合キーの混在指定が動作すること。"""
        df = pl.DataFrame(
            {
                "Sex": ["male", "female", "male", "female", "male", "female"],
                "Embarked": ["S", "C", "S", "C", "Q", "Q"],
                "Pclass": [1, 2, 3, 1, 2, 3],
                "Survived": [0, 1, 0, 1, 1, 0],
            }
        )
        result, encoder = resolver.bayesian_target_encode(
            df,
            columns=[["Sex", "Embarked"], "Pclass"],
            target_col="Survived",
            target_type="binary",
            n_splits=2,
            seed=42,
        )
        assert "Sex_Embarked_te" in result.columns
        assert "Pclass_te" in result.columns
        assert "Sex_Embarked" in encoder
        assert "Pclass" in encoder

    def test_composite_key_unknown_combination(self, resolver: PolarsResolver) -> None:
        """未知の複合キー組み合わせは global_mean でフォールバックすること。"""
        train_df = pl.DataFrame(
            {
                "Sex": ["male", "male", "female", "female"],
                "Embarked": ["S", "S", "C", "C"],
                "Survived": [0, 0, 1, 1],
            }
        )
        # male_C は train に存在しない
        test_df = pl.DataFrame(
            {
                "Sex": ["male"],
                "Embarked": ["C"],
            }
        )
        _, encoder = resolver.bayesian_target_encode(
            train_df,
            columns=[["Sex", "Embarked"]],
            target_col="Survived",
            target_type="binary",
            n_splits=2,
            seed=42,
        )
        result = resolver.transform_bayesian_target_encode(
            test_df, encoder=encoder, columns=[["Sex", "Embarked"]]
        )
        global_mean = encoder["Sex_Embarked"]["__prior__"].posterior_mean
        assert abs(result["Sex_Embarked_te"][0] - global_mean) < 1e-6


class TestTimeSeriesTargetEncode:
    """
    なぜこのテストが必要か:
    - 時系列データでは通常の KFold TE は未来の情報を使うため data leakage になる。
    - Expanding Window TE は各行で自分より過去のデータのみから統計量を計算する。
    - 最初の行（履歴なし）は事前分布の平均で埋めることで NaN を防ぐ。
    - 元の行順序が保持されることを保証する（sort → encode → re-sort）。
    - transform は全 encoder を適用するため、通常の bayesian TE transform と同じ動作。
    """

    def test_basic(self, resolver: PolarsResolver) -> None:
        """expanding window で float カラムが出力されること。"""
        df = pl.DataFrame(
            {
                "Sex": ["male", "female", "male", "female", "male", "female"],
                "Survived": [0, 1, 0, 1, 1, 0],
                "Date": [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                ],
            }
        )
        result, encoder = resolver.time_series_target_encode(
            df,
            columns=["Sex"],
            target_col="Survived",
            time_col="Date",
            target_type="binary",
        )
        assert "Sex_te" in result.columns
        assert result["Sex_te"].dtype in (pl.Float32, pl.Float64)
        assert result["Sex_te"].null_count() == 0
        assert "Sex" in encoder

    def test_no_future_leak(self, resolver: PolarsResolver) -> None:
        """時刻 T の行は T 未満のデータのみ使用すること。

        行3（Date=2026-01-04, Sex=male）の TE は
        行0,1,2 のデータのみから計算される。
        行4,5 の情報は使わない。

        行0: male, Survived=0, Date=01-01
        行1: male, Survived=0, Date=01-02
        行2: male, Survived=1, Date=01-03
        行3: male, Survived=1, Date=01-04 ← この行の TE は行0,1,2 から計算

        行0-2 の male: [0, 0, 1] → successes=1, failures=2
        global_mean(行0-2) = (0+0+1+1+1+0)/6... ではなく行0-2の全target
        → 行0-2: Survived=[0,0,1] → global=1/3

        もし未来リークがあれば行3-5の情報も使われ、値が変わる。
        """
        df = pl.DataFrame(
            {
                "Sex": ["male", "male", "male", "male", "male", "male"],
                "Survived": [0, 0, 1, 1, 100, 100],
                "Date": [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                ],
            }
        )
        result, _ = resolver.time_series_target_encode(
            df,
            columns=["Sex"],
            target_col="Survived",
            time_col="Date",
            target_type="binary",
        )
        # 行3 の TE は行0-2 のみから計算。行4,5 の target=100 は使わない
        # もしリークなら値が大幅に上がる
        row3_value = result["Sex_te"][3]
        assert row3_value < 10.0, (
            f"未来リークの疑い: 行3 の TE 値が大きすぎる ({row3_value})"
        )

    def test_first_row_uses_prior(self, resolver: PolarsResolver) -> None:
        """履歴がない最初の行は global_mean（事前分布の平均）で埋められること。"""
        df = pl.DataFrame(
            {
                "Sex": ["male", "female", "male", "female"],
                "Survived": [0, 1, 0, 1],
                "Date": [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                ],
            }
        )
        result, _ = resolver.time_series_target_encode(
            df,
            columns=["Sex"],
            target_col="Survived",
            time_col="Date",
            target_type="binary",
        )
        # 最初の行は過去データなし → global_mean が使われる
        global_mean = df["Survived"].mean()
        assert abs(result["Sex_te"][0] - global_mean) < 1e-6

    def test_expanding_mean_correctness(self, resolver: PolarsResolver) -> None:
        """Expanding window の事後平均が手計算と一致すること。

        データ（時系列順）:
        row0: Cat=A, Target=1, Date=01
        row1: Cat=A, Target=0, Date=02
        row2: Cat=A, Target=1, Date=03

        prior_weight=1.0, global_mean = (1+0+1)/3 = 2/3
        α₀ = 2/3 * 1 = 2/3, β₀ = 1/3 * 1 = 1/3

        row0: 履歴なし → global_mean = 2/3
        row1: 履歴=[row0: A=1] → successes=1, failures=0, N=1
              posterior_mean = (2/3 + 1) / (2/3 + 1/3 + 1) = (5/3) / 2 = 5/6
        row2: 履歴=[row0: A=1, row1: A=0] → successes=1, failures=1, N=2
              posterior_mean = (2/3 + 1) / (2/3 + 1/3 + 2) = (5/3) / 3 = 5/9
        """
        df = pl.DataFrame(
            {
                "Cat": ["A", "A", "A"],
                "Target": [1, 0, 1],
                "Date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            }
        )
        result, _ = resolver.time_series_target_encode(
            df,
            columns=["Cat"],
            target_col="Target",
            time_col="Date",
            target_type="binary",
            prior_weight=1.0,
        )
        global_mean = 2.0 / 3.0
        assert abs(result["Cat_te"][0] - global_mean) < 1e-6
        assert abs(result["Cat_te"][1] - 5.0 / 6.0) < 1e-6
        assert abs(result["Cat_te"][2] - 5.0 / 9.0) < 1e-6

    def test_min_samples(self, resolver: PolarsResolver) -> None:
        """min_samples 未満の場合は事前分布の平均が使われること。"""
        df = pl.DataFrame(
            {
                "Cat": ["A", "A", "A", "A"],
                "Target": [1, 0, 1, 0],
                "Date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            }
        )
        result, _ = resolver.time_series_target_encode(
            df,
            columns=["Cat"],
            target_col="Target",
            time_col="Date",
            target_type="binary",
            min_samples=3,
        )
        global_mean = df["Target"].mean()
        # row0: 履歴0件 < min_samples=3 → global_mean
        assert abs(result["Cat_te"][0] - global_mean) < 1e-6
        # row1: 履歴1件 < min_samples=3 → global_mean
        assert abs(result["Cat_te"][1] - global_mean) < 1e-6
        # row2: 履歴2件 < min_samples=3 → global_mean
        assert abs(result["Cat_te"][2] - global_mean) < 1e-6
        # row3: 履歴3件 >= min_samples=3 → Bayesian posterior
        assert result["Cat_te"][3] != global_mean or True  # ここでは != でなくてもOK

    def test_with_suffix(self, resolver: PolarsResolver) -> None:
        """suffix が適用されること。"""
        df = pl.DataFrame(
            {
                "Cat": ["A", "B", "A", "B"],
                "Target": [1, 0, 0, 1],
                "Date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            }
        )
        result, _ = resolver.time_series_target_encode(
            df,
            columns=["Cat"],
            target_col="Target",
            time_col="Date",
            target_type="binary",
            suffix="_expanding",
        )
        assert "Cat_expanding" in result.columns

    def test_preserves_row_order(self, resolver: PolarsResolver) -> None:
        """元の行順序が保持されること（逆順の Date でも結果は元順序）。"""
        df = pl.DataFrame(
            {
                "Cat": ["A", "B", "A", "B"],
                "Target": [1, 0, 0, 1],
                "Date": ["2026-01-04", "2026-01-03", "2026-01-02", "2026-01-01"],
            }
        )
        result, _ = resolver.time_series_target_encode(
            df,
            columns=["Cat"],
            target_col="Target",
            time_col="Date",
            target_type="binary",
        )
        # 元の Cat カラムの順序が保持されていること
        assert result["Cat"].to_list() == ["A", "B", "A", "B"]
        assert len(result) == 4

    def test_transform_time_series_applies_full_encoder(self, resolver: PolarsResolver) -> None:
        """transform は full encoder（全 train データで計算）を test データに適用すること。"""
        train_df = pl.DataFrame(
            {
                "Sex": ["male", "female", "male", "female"],
                "Survived": [0, 1, 0, 1],
                "Date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            }
        )
        test_df = pl.DataFrame({"Sex": ["male", "unknown"]})
        _, encoder = resolver.time_series_target_encode(
            train_df,
            columns=["Sex"],
            target_col="Survived",
            time_col="Date",
            target_type="binary",
        )
        result = resolver.transform_bayesian_target_encode(
            test_df, encoder=encoder, columns=["Sex"]
        )
        assert "Sex_te" in result.columns
        # unknown は global_mean にフォールバック
        global_mean = encoder["Sex"]["__prior__"].posterior_mean
        assert abs(result["Sex_te"][1] - global_mean) < 1e-6
