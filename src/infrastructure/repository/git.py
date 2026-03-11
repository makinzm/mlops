"""
Git リポジトリ操作の実装。
"""

import subprocess
from pathlib import Path

_DATA_DIR_GITIGNORE = """\
*
!.gitignore
!.gitkeep
!metadata.yaml
"""


class GitRepositoryImpl:
    def get_commit_hash(self) -> str:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    def setup_data_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / ".gitkeep").touch()
        gitignore = path / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(_DATA_DIR_GITIGNORE)
