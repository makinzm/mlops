"""Kaggle API 認証ヘルパー。

main.py に 3 箇所重複していた認証ボイラープレートを集約する。
kaggle パッケージは import 時に authenticate() を実行するため、
import 文自体を try/except SystemExit で包む必要がある。
"""

from __future__ import annotations

from typing import Any


def authenticate_kaggle_api() -> Any:
    """認証済み KaggleApi インスタンスを返す。

    Returns:
        KaggleApi: 認証済みインスタンス

    Raises:
        RuntimeError: 認証に失敗した場合（SystemExit を変換）
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except SystemExit as e:
        raise RuntimeError(
            "Kaggle 認証に失敗しました。~/.kaggle/access_token を確認してください。"
        ) from e

    api = KaggleApi()
    try:
        api.authenticate()
    except SystemExit as e:
        raise RuntimeError(
            "Kaggle 認証に失敗しました。~/.kaggle/access_token を確認してください。"
        ) from e

    return api
