"""
Git リポジトリ操作のドメイン定義。

Git に関する操作はすべてこの Protocol を経由することで、
インフラ実装（subprocess 等）への直接依存を排除する。
"""

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class GitRepository(Protocol):
    def get_commit_hash(self) -> str:
        """現在の HEAD の commit hash を返す。再現性記録のために使用する。"""
        ...

    def setup_data_dir(self, path: Path) -> None:
        """データ出力ディレクトリを作成し、git 管理の初期化を行う。

        作成するファイル:
        - .gitkeep: 空ディレクトリを git 追跡させるため
        - .gitignore: データファイルを無視しつつ .gitkeep と .gitignore 自体は残す
        """
        ...
