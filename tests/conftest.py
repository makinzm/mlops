"""
pytest 設定。

なぜこのファイルが必要か:
- `integration` マーカーを登録し、CI では実認証が必要なテストをスキップできるようにする。
- kaggle パッケージは import 時に authenticate() を呼び出し、認証情報がない場合 exit(1) する。
  sys.modules に事前にモックを登録することで、CI 環境でも kaggle を import できるようにする。
"""

import sys
from unittest.mock import MagicMock

import pytest

# kaggle/__init__.py が import 時に api.authenticate() → exit(1) を呼ぶのを防ぐ。
# patch() が対象モジュールを解決する際にも import が走るため、
# テスト収集前に sys.modules に登録しておく必要がある。
if "kaggle" not in sys.modules:
    _mock_api_module = MagicMock()
    _mock_api_module.KaggleApi = MagicMock
    sys.modules["kaggle"] = MagicMock()
    sys.modules["kaggle.api"] = MagicMock()
    sys.modules["kaggle.api.kaggle_api_extended"] = _mock_api_module

# google-cloud-* は GCP 認証情報がない CI 環境で import 時に失敗する可能性があるため、
# テスト収集前に sys.modules にモックを登録しておく。
# テスト内では patch() でさらに上書きして動作を制御する。
if "google" not in sys.modules:
    sys.modules["google"] = MagicMock()
    sys.modules["google.cloud"] = MagicMock()
    sys.modules["google.cloud.storage"] = MagicMock()
    sys.modules["google.cloud.aiplatform"] = MagicMock()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring real Kaggle credentials (skip in CI)",
    )
