import os
from pathlib import Path

# どのディレクトリから uv run python -m src を実行しても
# conf/ や data/ の相対パスがプロジェクトルートから解決されるように chdir する。
# src/__main__.py の親の親 = プロジェクトルート
os.chdir(Path(__file__).parent.parent)

from src.main import main  # noqa: E402

main()
