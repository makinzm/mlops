"""
pytest 設定。

なぜこのファイルが必要か:
- `integration` マーカーを登録し、CI では実認証が必要なテストをスキップできるようにする。
- `uv run pytest -m "not integration"` で CI 用モックテストのみ実行できる。
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring real Kaggle credentials (skip in CI)",
    )
