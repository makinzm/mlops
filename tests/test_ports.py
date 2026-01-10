"""Portsのインターフェーステスト"""

from pathlib import Path

import pandas as pd

from src.adapters.local import LocalDataStore, LocalModelStore


def test_local_data_store(tmp_path: Path) -> None:
    store = LocalDataStore(tmp_path)
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    path = store.save_processed(df, "test")
    assert path.exists()

    loaded = store.load_processed("test")
    assert loaded.equals(df)


def test_local_model_store(tmp_path: Path) -> None:
    store = LocalModelStore(tmp_path)
    model = {"weights": [1, 2, 3]}

    store.save(model, "test_model")
    loaded = store.load("test_model")

    assert loaded == model
    assert "test_model" in store.list_models()
