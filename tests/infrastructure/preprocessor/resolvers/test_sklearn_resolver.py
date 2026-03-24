"""
SklearnResolver の fill_na / target_encode メソッドに対するユニットテスト。

なぜこのテストが必要か（fill_na）:
- fill_na は Train データで fit した統計量（median/mean/constant）を
  Test データに transform することで、データリークを防ぐ設計になっている。
- 特に「Test データの median ではなく Train の median で埋める」ことを
  明示的に検証する必要がある。これがないと将来の変更でリークが生じても検知できない。
- sklearn の SimpleImputer の fit/transform 分離を wrapper として正しく実装できているかを確認する。

なぜこのテストが必要か（target_encode）:
- Target Encoding は OOF（Out-of-Fold）CV で実装する必要がある。
  同一 fold の val データを encoder の fit に使うと target leakage が発生するため。
- Test 時は全 train データで fit した encoder を適用する必要がある。
- smoothing は小カテゴリの過学習を防ぐが、その効果をテストで明示的に確認しないと
  将来の実装変更でサイレントに壊れる恐れがある。
- 未知カテゴリ（Test のみに現れる値）が KeyError にならず global_mean で
  フォールバックされることの確認が必要。
"""

import polars as pl
import pytest

from src.infrastructure.preprocessor.resolvers.sklearn_resolver import SklearnResolver


@pytest.fixture()
def resolver() -> SklearnResolver:
    return SklearnResolver()


class TestFillNaMedian:
    def test_fill_na_fills_nulls(self, resolver: SklearnResolver) -> None:
        """median 戦略で null が埋められること。"""
        train_df = pl.DataFrame({"col1": [1.0, 2.0, None, 4.0, 5.0]})
        result_train, _ = resolver.fill_na(train_df, strategy="median", columns=["col1"])
        assert result_train["col1"].null_count() == 0

    def test_fill_na_median_value(self, resolver: SklearnResolver) -> None:
        """null が Train データの median（3.0）で埋められること。"""
        train_df = pl.DataFrame({"col1": [1.0, 2.0, None, 4.0, 5.0]})
        result_train, _ = resolver.fill_na(train_df, strategy="median", columns=["col1"])
        # null だった箇所が median=3.0 で埋まること（null が index=2 にある）
        # 全体の median は [1,2,4,5] の中央値 = (2+4)/2 = 3.0
        assert result_train["col1"][2] == 3.0

    def test_fill_na_no_data_leak(self, resolver: SklearnResolver) -> None:
        """Train の統計量が Test に適用されること（データリーク防止）。

        Train: [1, 2, 3] → median = 2.0
        Test:  [10, None, 30]
        → Test の null は Train の median (2.0) で埋まるべき。
           Test 自体の median (20.0) で埋まってはいけない。
        """
        train_df = pl.DataFrame({"col1": [1.0, 2.0, 3.0]})
        test_df = pl.DataFrame({"col1": [10.0, None, 30.0]})
        _, imputer = resolver.fill_na(train_df, strategy="median", columns=["col1"])
        result_test = resolver.transform(test_df, imputer=imputer, columns=["col1"])
        # Train median は 2.0、Test 単体の median は 20.0
        assert result_test["col1"][1] == 2.0

    def test_fill_na_mean_strategy(self, resolver: SklearnResolver) -> None:
        """mean 戦略で null が Train の mean で埋められること。"""
        train_df = pl.DataFrame({"col1": [1.0, 2.0, None, 4.0]})
        result_train, _ = resolver.fill_na(train_df, strategy="mean", columns=["col1"])
        # mean of [1, 2, 4] = 7/3 ≈ 2.333...
        filled_val = result_train["col1"][2]
        assert filled_val is not None
        assert abs(filled_val - (1.0 + 2.0 + 4.0) / 3.0) < 1e-6

    def test_fill_na_constant_strategy(self, resolver: SklearnResolver) -> None:
        """constant 戦略で null が指定値（-999）で埋められること。"""
        train_df = pl.DataFrame({"col1": [1.0, None, 3.0]})
        result_train, _ = resolver.fill_na(
            train_df, strategy="constant", columns=["col1"], fill_value=-999.0
        )
        assert result_train["col1"][1] == -999.0

    def test_fill_na_multiple_columns(self, resolver: SklearnResolver) -> None:
        """複数カラムを同時に補完できること。"""
        train_df = pl.DataFrame(
            {
                "col1": [1.0, None, 3.0],
                "col2": [None, 2.0, 3.0],
                "label": [0, 1, 0],
            }
        )
        result_train, _ = resolver.fill_na(train_df, strategy="median", columns=["col1", "col2"])
        assert result_train["col1"].null_count() == 0
        assert result_train["col2"].null_count() == 0
        # label カラムは変更されない
        assert result_train["label"].to_list() == [0, 1, 0]

    def test_supported_methods_includes_fill_na(self, resolver: SklearnResolver) -> None:
        """supported_methods() に fill_na が含まれること。"""
        assert "fill_na" in resolver.supported_methods()


class TestTargetEncode:
    """
    なぜこのテストが必要か:
    - Target Encoding の OOF 実装は「val fold のデータを encoder の fit に使わない」ことが核心。
      これをテストで明示的に検証しないと、実装の変更でリークが生じても気づけない。
    - smoothing=0.0 と smoothing>0 の差を数値で確認することで、
      smoothing が実際に機能していることを保証できる。
    - 未知カテゴリの global_mean フォールバックは推論時に必ず発生しうる。
      KeyError でクラッシュしないことをテストで保証する。
    """

    def test_target_encode_oof_no_leak(self, resolver: SklearnResolver) -> None:
        """OOF encode で val fold の行が fit 側の統計量に影響しないこと。

        fold 0 の val 行（index=0,1）に大きな target 値（100.0）を置くと、
        もしリークがあれば val 行の encoded 値が非常に大きくなる。
        OOF が正しく実装されていれば、fold 0 の encoder は
        index=2,3 の train 行のみで fit されるため val の値に引きずられない。

        fixture:
          rows 0-1: val fold (Survived=100.0 — 意図的に外れ値)
          rows 2-3: train fold (Survived=0.0)
          Sex は全行 "male"
          n_splits=2 で 0-1 が fold 0 val, 2-3 が fold 0 train
        """
        df = pl.DataFrame(
            {
                "Sex": ["male", "male", "male", "male"],
                "Survived": [100.0, 100.0, 0.0, 0.0],
            }
        )
        # n_splits=2, seed=0 → fold 0: train=[2,3] val=[0,1]
        result, _ = resolver.target_encode(
            df, columns=["Sex"], target_col="Survived", n_splits=2, seed=0
        )
        # val 行(0,1)の encoded 値は fold 0 の train([2,3]) の Sex mean で計算される。
        # train[2,3] の Survived mean = 0.0
        # smoothing=1.0: encoded = (2*0.0 + 1*0.0) / (2+1) = 0.0
        # リークがあれば val 行は 100.0 に近くなる
        val_values = result["Sex"].to_list()[:2]
        assert all(v is not None for v in val_values)
        # val 行の encoded 値が 50.0 を超えていればリーク（正常なら 0 に近い値）
        assert all(v < 50.0 for v in val_values), (
            f"OOF リークの疑い: val 行の encoded 値が大きすぎる {val_values}"
        )

    def test_target_encode_replaces_category_with_mean(self, resolver: SklearnResolver) -> None:
        """カテゴリが target 平均値に置換されること（基本動作確認）。

        n_splits=2 で全行を OOF encode する。
        結果は float カラムになり、null がないこと。
        """
        df = pl.DataFrame(
            {
                "Sex": ["male", "female", "male", "female"],
                "Survived": [0, 1, 0, 1],
            }
        )
        result, full_encoder = resolver.target_encode(
            df, columns=["Sex"], target_col="Survived", n_splits=2, seed=42
        )
        assert result["Sex"].dtype in (pl.Float32, pl.Float64)
        assert result["Sex"].null_count() == 0
        # full_encoder は全 train で fit した encoder
        assert "Sex" in full_encoder
        assert "__global_mean__" in full_encoder["Sex"]

    def test_target_encode_unknown_category_uses_global_mean(
        self, resolver: SklearnResolver
    ) -> None:
        """Test の未知カテゴリが KeyError でなく global_mean でフォールバックされること。

        Train: Sex = [male, female] のみ
        Test:  Sex = ["unknown"] という Train に存在しない値
        → KeyError を起こさず global_mean で埋めること。
        """
        train_df = pl.DataFrame(
            {
                "Sex": ["male", "female", "male", "female"],
                "Survived": [0, 1, 0, 1],
            }
        )
        test_df = pl.DataFrame({"Sex": ["unknown_category"]})

        _, full_encoder = resolver.target_encode(
            train_df, columns=["Sex"], target_col="Survived", n_splits=2, seed=42
        )
        # KeyError が起きないこと
        result_test = resolver.transform_target_encode(
            test_df, encoder=full_encoder, columns=["Sex"]
        )
        global_mean = full_encoder["Sex"]["__global_mean__"]
        assert abs(result_test["Sex"][0] - global_mean) < 1e-6

    def test_target_encode_smoothing_applied(self, resolver: SklearnResolver) -> None:
        """smoothing > 0 のとき小カテゴリの値が global_mean 寄りになること。

        Train: Sex=[male(1件), female(3件)], Survived=[1,0,0,0]
        global_mean = 0.25、male の raw mean = 1.0
        smoothing=0.0: encoded = 1.0（raw mean そのまま）
        smoothing=10.0: encoded = (1*1.0 + 10*0.25)/(1+10) ≈ 0.318（global_mean 寄り）
        → smoothing=10 の方が male の encoded 値が小さい。
        """
        train_df = pl.DataFrame(
            {
                "Sex": ["male", "female", "female", "female"],
                "Survived": [1, 0, 0, 0],
            }
        )
        # n_splits=2, male は index=0 が入る fold の val になる
        # smoothing=0.0 vs smoothing=10.0 で結果が変わることを確認する
        # full_encoder（全 train）を使って test データで比較する
        test_df = pl.DataFrame({"Sex": ["male"]})

        _, encoder_no_smooth = resolver.target_encode(
            train_df, columns=["Sex"], target_col="Survived", n_splits=2, seed=42, smoothing=0.0
        )
        _, encoder_with_smooth = resolver.target_encode(
            train_df, columns=["Sex"], target_col="Survived", n_splits=2, seed=42, smoothing=10.0
        )
        result_no_smooth = resolver.transform_target_encode(
            test_df, encoder=encoder_no_smooth, columns=["Sex"]
        )
        result_with_smooth = resolver.transform_target_encode(
            test_df, encoder=encoder_with_smooth, columns=["Sex"]
        )
        # smoothing あり → global_mean(0.25) 寄りになるため no_smooth(≈1.0) より小さい値
        assert result_no_smooth["Sex"][0] > result_with_smooth["Sex"][0]

    def test_target_encode_multiple_columns(self, resolver: SklearnResolver) -> None:
        """複数カラムを同時にエンコードできること。"""
        df = pl.DataFrame(
            {
                "Sex": ["male", "female", "male", "female"],
                "Embarked": ["S", "C", "S", "Q"],
                "Survived": [0, 1, 0, 1],
            }
        )
        result, full_encoder = resolver.target_encode(
            df, columns=["Sex", "Embarked"], target_col="Survived", n_splits=2, seed=42
        )
        assert result["Sex"].dtype in (pl.Float32, pl.Float64)
        assert result["Embarked"].dtype in (pl.Float32, pl.Float64)
        assert "Sex" in full_encoder
        assert "Embarked" in full_encoder

    def test_transform_target_encode_applies_train_encoder_to_test(
        self, resolver: SklearnResolver
    ) -> None:
        """train で fit した full_encoder を test データに適用できること。

        Train: male の Survived mean = 0.0（全行 male, Survived=0）
        Test:  male のみ
        → Test の male は Train の統計量（≈0.0）でエンコードされること。
           Test 独自の mean ではないことを確認する。
        """
        train_df = pl.DataFrame(
            {
                "Sex": ["male", "male", "male", "male"],
                "Survived": [0, 0, 0, 0],
            }
        )
        test_df = pl.DataFrame({"Sex": ["male"]})

        _, full_encoder = resolver.target_encode(
            train_df, columns=["Sex"], target_col="Survived", n_splits=2, seed=42
        )
        result_test = resolver.transform_target_encode(
            test_df, encoder=full_encoder, columns=["Sex"]
        )
        # Train の male mean = 0.0 → smoothing=1, global_mean=0.0
        # encoded = (4*0 + 1*0) / (4+1) = 0.0
        assert result_test["Sex"].dtype in (pl.Float32, pl.Float64)
        assert abs(result_test["Sex"][0] - 0.0) < 1e-6

    def test_supported_methods_includes_target_encode(self, resolver: SklearnResolver) -> None:
        """supported_methods() に target_encode が含まれること。"""
        assert "target_encode" in resolver.supported_methods()


class TestTargetEncodeSuffix:
    """
    なぜこのテストが必要か:
    - 既存の target_encode は元カラム名を上書きするため、
      前処理後に元のカテゴリカラムと TE カラムを区別できない。
    - suffix/prefix を追加することで `Sex` → `Sex_te` のように区別可能にする。
    - デフォルト（空文字）で後方互換を保証し、既存パイプラインが壊れないことを確認する。
    - encoder dict のキーは元カラム名のまま保持し、transform 時に suffix を適用する。
    """

    def test_suffix_renames_columns(self, resolver: SklearnResolver) -> None:
        """suffix="_te" で Sex_te カラムが生成されること。

        suffix 指定時は元カラム Sex は保持されたまま、
        新しいカラム Sex_te が追加される。
        """
        df = pl.DataFrame(
            {
                "Sex": ["male", "female", "male", "female"],
                "Survived": [0, 1, 0, 1],
            }
        )
        result, _ = resolver.target_encode(
            df, columns=["Sex"], target_col="Survived", n_splits=2, seed=42, suffix="_te"
        )
        assert "Sex_te" in result.columns
        assert "Sex" in result.columns
        assert result["Sex_te"].dtype in (pl.Float32, pl.Float64)
        assert result["Sex_te"].null_count() == 0

    def test_prefix_renames_columns(self, resolver: SklearnResolver) -> None:
        """prefix="enc_" で enc_Sex カラムが生成されること。"""
        df = pl.DataFrame(
            {
                "Sex": ["male", "female", "male", "female"],
                "Survived": [0, 1, 0, 1],
            }
        )
        result, _ = resolver.target_encode(
            df, columns=["Sex"], target_col="Survived", n_splits=2, seed=42, prefix="enc_"
        )
        assert "enc_Sex" in result.columns
        assert "Sex" in result.columns

    def test_suffix_and_prefix_combined(self, resolver: SklearnResolver) -> None:
        """prefix="te_" + suffix="_v1" で te_Sex_v1 カラムが生成されること。"""
        df = pl.DataFrame(
            {
                "Sex": ["male", "female", "male", "female"],
                "Survived": [0, 1, 0, 1],
            }
        )
        result, _ = resolver.target_encode(
            df,
            columns=["Sex"],
            target_col="Survived",
            n_splits=2,
            seed=42,
            suffix="_v1",
            prefix="te_",
        )
        assert "te_Sex_v1" in result.columns
        assert "Sex" in result.columns

    def test_no_suffix_backward_compatible(self, resolver: SklearnResolver) -> None:
        """デフォルト（suffix/prefix なし）で元カラム上書きの既存動作が維持されること。"""
        df = pl.DataFrame(
            {
                "Sex": ["male", "female", "male", "female"],
                "Survived": [0, 1, 0, 1],
            }
        )
        result, _ = resolver.target_encode(
            df, columns=["Sex"], target_col="Survived", n_splits=2, seed=42
        )
        # suffix/prefix なし → 元カラムが float に上書きされる（既存動作）
        assert result["Sex"].dtype in (pl.Float32, pl.Float64)
        assert result["Sex"].null_count() == 0

    def test_transform_suffix_applied(self, resolver: SklearnResolver) -> None:
        """transform_target_encode でも suffix が適用されること。"""
        train_df = pl.DataFrame(
            {
                "Sex": ["male", "female", "male", "female"],
                "Survived": [0, 1, 0, 1],
            }
        )
        test_df = pl.DataFrame({"Sex": ["male", "female"]})
        _, full_encoder = resolver.target_encode(
            train_df, columns=["Sex"], target_col="Survived", n_splits=2, seed=42, suffix="_te"
        )
        result = resolver.transform_target_encode(
            test_df, encoder=full_encoder, columns=["Sex"], suffix="_te"
        )
        assert "Sex_te" in result.columns
        assert "Sex" in result.columns
        assert result["Sex_te"].dtype in (pl.Float32, pl.Float64)

    def test_encoder_keys_unchanged(self, resolver: SklearnResolver) -> None:
        """suffix 指定しても encoder dict のキーは元カラム名のままであること。

        encoder のキーが変わると transform 時にキー不一致で壊れるため、
        suffix はカラム名出力のみに影響し、encoder 内部は変更しない。
        """
        df = pl.DataFrame(
            {
                "Sex": ["male", "female", "male", "female"],
                "Survived": [0, 1, 0, 1],
            }
        )
        _, encoder = resolver.target_encode(
            df, columns=["Sex"], target_col="Survived", n_splits=2, seed=42, suffix="_te"
        )
        assert "Sex" in encoder
        assert "Sex_te" not in encoder
