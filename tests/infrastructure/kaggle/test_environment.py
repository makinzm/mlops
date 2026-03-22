"""
KaggleEnvironment のテスト。

Kaggle Notebook 環境の検出とパス解決が正しく動作することを検証する。
- is_kaggle_notebook(): KAGGLE_KERNEL_RUN_TYPE 環境変数の有無で判定
- resolve_input_root(slug): Kaggle なら /kaggle/input/{slug}、ローカルなら Path(slug)
- resolve_output_root(): Kaggle なら /kaggle/working、ローカルなら Path(".")

環境変数の操作は monkeypatch で行い、テスト間の副作用を防ぐ。
"""

from pathlib import Path

import pytest

from src.infrastructure.kaggle.environment import KaggleEnvironment

_ENV_KEY = "KAGGLE_KERNEL_RUN_TYPE"


class TestIsKaggleNotebook:
    """is_kaggle_notebook() の判定ロジックを検証する。"""

    def test_is_kaggle_notebook_false_when_env_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """KAGGLE_KERNEL_RUN_TYPE が未設定のときローカル環境と判定する。"""
        monkeypatch.delenv(_ENV_KEY, raising=False)
        assert KaggleEnvironment.is_kaggle_notebook() is False

    def test_is_kaggle_notebook_true_when_env_interactive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """KAGGLE_KERNEL_RUN_TYPE=Interactive のとき Kaggle 環境と判定する。"""
        monkeypatch.setenv(_ENV_KEY, "Interactive")
        assert KaggleEnvironment.is_kaggle_notebook() is True

    def test_is_kaggle_notebook_true_when_env_batch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """KAGGLE_KERNEL_RUN_TYPE=Batch のとき Kaggle 環境と判定する。"""
        monkeypatch.setenv(_ENV_KEY, "Batch")
        assert KaggleEnvironment.is_kaggle_notebook() is True


class TestResolveInputRoot:
    """resolve_input_root(slug) のパス解決を検証する。"""

    def test_resolve_input_root_local(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ローカル環境では slug をそのまま Path に変換する。"""
        monkeypatch.delenv(_ENV_KEY, raising=False)
        result = KaggleEnvironment.resolve_input_root("titanic")
        assert result == Path("titanic")

    def test_resolve_input_root_kaggle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Kaggle 環境では /kaggle/input/competitions/{slug} を返す。"""
        monkeypatch.setenv(_ENV_KEY, "Interactive")
        result = KaggleEnvironment.resolve_input_root("titanic")
        assert result == Path("/kaggle/input/competitions/titanic")


class TestResolveOutputRoot:
    """resolve_output_root() のパス解決を検証する。"""

    def test_resolve_output_root_local(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ローカル環境では Path(".") を返す。"""
        monkeypatch.delenv(_ENV_KEY, raising=False)
        result = KaggleEnvironment.resolve_output_root()
        assert result == Path(".")

    def test_resolve_output_root_kaggle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Kaggle 環境では /kaggle/working を返す。"""
        monkeypatch.setenv(_ENV_KEY, "Interactive")
        result = KaggleEnvironment.resolve_output_root()
        assert result == Path("/kaggle/working")
