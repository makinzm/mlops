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
        with patch(
            "src.presentation.kaggle_auth.KaggleApi", return_value=mock_api
        ):
            from src.presentation.kaggle_auth import authenticate_kaggle_api

            result = authenticate_kaggle_api()

        assert result is mock_api
        mock_api.authenticate.assert_called_once()

    def test_raises_runtime_error_on_import_system_exit(self) -> None:
        """import 時の SystemExit が RuntimeError に変換されること。"""
        from src.presentation.kaggle_auth import authenticate_kaggle_api

        with (
            patch(
                "src.presentation.kaggle_auth.KaggleApi",
                side_effect=SystemExit(1),
            ),
            pytest.raises(RuntimeError, match="Kaggle 認証に失敗"),
        ):
            authenticate_kaggle_api()

    def test_raises_runtime_error_on_authenticate_system_exit(self) -> None:
        """authenticate() 時の SystemExit が RuntimeError に変換されること。"""
        mock_api = MagicMock()
        mock_api.authenticate.side_effect = SystemExit(1)

        with (
            patch(
                "src.presentation.kaggle_auth.KaggleApi",
                return_value=mock_api,
            ),
            pytest.raises(RuntimeError, match="Kaggle 認証に失敗"),
        ):
            from src.presentation.kaggle_auth import authenticate_kaggle_api

            authenticate_kaggle_api()
