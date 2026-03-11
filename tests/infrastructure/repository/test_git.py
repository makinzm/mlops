"""
GitRepositoryImpl のテスト。

なぜこのテストが必要か:
- setup_data_dir が .gitkeep と .gitignore を正しく作成することを保証する。
- 出力ディレクトリが conf で自由に変えられるため、どのパスでも正しく機能することを確認する。
- .gitignore の内容がデータファイルを無視しつつ .gitkeep を残す設定になっていることを保証する。
- get_commit_hash がリポジトリ内で正常に動作することを確認する。
"""

from pathlib import Path

from src.infrastructure.repository.git import GitRepositoryImpl


class TestGitRepositoryImplSetupDataDir:
    def test_creates_directory(self, tmp_path: Path) -> None:
        """setup_data_dir が指定パスのディレクトリを作成すること。

        conf で任意の output_dir が設定されるため、存在しないパスでも作成できることを保証する。
        """
        target = tmp_path / "nested" / "output"
        GitRepositoryImpl().setup_data_dir(target)
        assert target.is_dir()

    def test_creates_gitkeep(self, tmp_path: Path) -> None:
        """.gitkeep が作成され、空ディレクトリが git 追跡できること。"""
        GitRepositoryImpl().setup_data_dir(tmp_path)
        assert (tmp_path / ".gitkeep").exists()

    def test_creates_gitignore(self, tmp_path: Path) -> None:
        """.gitignore が作成されること。"""
        GitRepositoryImpl().setup_data_dir(tmp_path)
        assert (tmp_path / ".gitignore").exists()

    def test_gitignore_ignores_data_files(self, tmp_path: Path) -> None:
        """.gitignore がデータファイルを無視しつつ .gitkeep と .gitignore 自体は残す内容であること。

        ダウンロードした大容量ファイルが誤って git add されるのを防ぐ。
        """
        GitRepositoryImpl().setup_data_dir(tmp_path)
        content = (tmp_path / ".gitignore").read_text()
        assert "*" in content
        assert "!.gitignore" in content
        assert "!.gitkeep" in content

    def test_does_not_overwrite_existing_gitignore(self, tmp_path: Path) -> None:
        """既存の .gitignore を上書きしないこと。

        ユーザーが手動でカスタマイズした .gitignore を保護する。
        """
        custom = "custom content"
        (tmp_path / ".gitignore").write_text(custom)
        GitRepositoryImpl().setup_data_dir(tmp_path)
        assert (tmp_path / ".gitignore").read_text() == custom

    def test_idempotent(self, tmp_path: Path) -> None:
        """複数回呼び出しても冪等であること。

        ダウンロードを再実行した場合でも正常に動作することを保証する。
        """
        repo = GitRepositoryImpl()
        repo.setup_data_dir(tmp_path)
        repo.setup_data_dir(tmp_path)  # 2回目
        assert (tmp_path / ".gitkeep").exists()


class TestGitRepositoryImplGetCommitHash:
    def test_returns_string(self) -> None:
        """get_commit_hash が文字列を返すこと。"""
        result = GitRepositoryImpl().get_commit_hash()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_hex_or_unknown(self) -> None:
        """git リポジトリ内では 40 文字の hex か 'unknown' を返すこと。"""
        result = GitRepositoryImpl().get_commit_hash()
        is_hex = len(result) == 40 and all(c in "0123456789abcdef" for c in result)
        assert result == "unknown" or is_hex
