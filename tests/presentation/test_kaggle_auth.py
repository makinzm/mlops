"""
presentation/kaggle_auth モジュールのテスト。

なぜこのテストが必要か:
  - Kaggle 認証ボイラープレートが main.py に 3 箇所重複していた。
  - authenticate_kaggle_api() に集約したことで、認証ロジックの一貫性を保証する。
  - SystemExit が RuntimeError に変換されることを検証する。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestAuthenticateKaggleApi:
    """authenticate_kaggle_api() のテスト。"""

    def test_returns_authenticated_api(self) -> None:
        """正常系: 認証済み KaggleApi インスタンスを返すこと。"""
        mock_api = MagicMock()
        with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=mock_api):
            from src.presentation.kaggle_auth import authenticate_kaggle_api

            result = authenticate_kaggle_api()

        assert result is mock_api
        mock_api.authenticate.assert_called_once()

    def test_raises_runtime_error_on_import_system_exit(self) -> None:
        """import 時の SystemExit が RuntimeError に変換されること。

        kaggle パッケージは import 時に authenticate() を実行するため、
        import 文自体で SystemExit が発生するケースをシミュレートする。
        builtins.__import__ をパッチして kaggle モジュールの import で SystemExit を発生させる。
        """
        import builtins
        import sys

        from src.presentation.kaggle_auth import authenticate_kaggle_api

        # kaggle モジュールをキャッシュから除去して再 import を強制
        modules_to_remove = [k for k in sys.modules if k.startswith("kaggle")]
        saved = {k: sys.modules.pop(k) for k in modules_to_remove}

        original_import = builtins.__import__

        def _raise_on_kaggle(name: str, *args: object, **kwargs: object) -> object:
            if name.startswith("kaggle"):
                raise SystemExit(1)
            return original_import(name, *args, **kwargs)  # ty:ignore[invalid-argument-type]

        try:
            with (
                patch("builtins.__import__", side_effect=_raise_on_kaggle),
                pytest.raises(RuntimeError, match="Kaggle 認証に失敗"),
            ):
                authenticate_kaggle_api()
        finally:
            # モジュールキャッシュを復元
            sys.modules.update(saved)

    def test_raises_runtime_error_on_authenticate_system_exit(self) -> None:
        """authenticate() 時の SystemExit が RuntimeError に変換されること。"""
        mock_api = MagicMock()
        mock_api.authenticate.side_effect = SystemExit(1)

        with (
            patch(
                "kaggle.api.kaggle_api_extended.KaggleApi",
                return_value=mock_api,
            ),
            pytest.raises(RuntimeError, match="Kaggle 認証に失敗"),
        ):
            from src.presentation.kaggle_auth import authenticate_kaggle_api

            authenticate_kaggle_api()
