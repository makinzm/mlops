"""
設定ファイルの妥当性を検証するテスト。

なぜこのテストが必要か:
- lefthook.yml と ci.yml はインフラ設定ファイルであり、存在しない・壊れている場合
  開発者が全く気づかずにリポジトリを使い続けるリスクがある。
- CI や pre-commit フックが正しいコマンドを持っているかをコードとして明示することで、
  設定の「仕様」をドキュメントではなくテストで表現できる。
- 設定変更時のリグレッション防止として機能する。
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
LEFTHOOK_YML = REPO_ROOT / "lefthook.yml"
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"

REQUIRED_LEFTHOOK_COMMANDS = {"ruff-check", "ruff-format", "mypy", "actionlint"}
REQUIRED_CI_JOBS = {"lint", "type-check", "test", "actionlint"}


class TestLefthookConfig:
    def test_lefthook_yml_exists(self) -> None:
        """lefthook.yml がリポジトリルートに存在する。"""
        assert LEFTHOOK_YML.exists(), f"{LEFTHOOK_YML} が存在しません"

    def test_lefthook_yml_is_valid_yaml(self) -> None:
        """lefthook.yml が有効な YAML としてパース可能である。"""
        content = LEFTHOOK_YML.read_text()
        parsed = yaml.safe_load(content)
        assert parsed is not None, "lefthook.yml が空または無効な YAML です"

    def test_lefthook_has_pre_commit_section(self) -> None:
        """lefthook.yml に pre-commit セクションが定義されている。"""
        parsed = yaml.safe_load(LEFTHOOK_YML.read_text())
        assert "pre-commit" in parsed, "pre-commit セクションが lefthook.yml に存在しません"

    def test_lefthook_pre_commit_has_required_commands(self) -> None:
        """pre-commit フックに必須コマンドがすべて定義されている。"""
        parsed = yaml.safe_load(LEFTHOOK_YML.read_text())
        commands = set(parsed["pre-commit"].get("commands", {}).keys())
        missing = REQUIRED_LEFTHOOK_COMMANDS - commands
        assert not missing, f"lefthook.yml に以下のコマンドが不足しています: {missing}"


class TestCIWorkflow:
    def test_ci_yml_exists(self) -> None:
        """.github/workflows/ci.yml が存在する。"""
        assert CI_YML.exists(), f"{CI_YML} が存在しません"

    def test_ci_yml_is_valid_yaml(self) -> None:
        """ci.yml が有効な YAML としてパース可能である。"""
        content = CI_YML.read_text()
        parsed = yaml.safe_load(content)
        assert parsed is not None, "ci.yml が空または無効な YAML です"

    def test_ci_has_jobs_section(self) -> None:
        """ci.yml に jobs セクションが定義されている。"""
        parsed = yaml.safe_load(CI_YML.read_text())
        assert "jobs" in parsed, "jobs セクションが ci.yml に存在しません"

    def test_ci_has_required_jobs(self) -> None:
        """CI ワークフローに必須ジョブがすべて定義されている。"""
        parsed = yaml.safe_load(CI_YML.read_text())
        jobs = set(parsed["jobs"].keys())
        missing = REQUIRED_CI_JOBS - jobs
        assert not missing, f"ci.yml に以下のジョブが不足しています: {missing}"

    def test_ci_triggers_on_main_push_and_pr(self) -> None:
        """CI は main への push と PR をトリガーとして設定されている。

        PyYAML は `on:` キーをブール値 True として解析するため、
        `"on"` と `True` の両方のキーを検索する。
        """
        parsed = yaml.safe_load(CI_YML.read_text())
        # PyYAML parses bare `on` as boolean True
        on = parsed.get("on", parsed.get(True, {}))
        assert "push" in on, "push トリガーが ci.yml に設定されていません"
        assert "pull_request" in on, "pull_request トリガーが ci.yml に設定されていません"
        push_branches = on["push"].get("branches", [])
        pr_branches = on["pull_request"].get("branches", [])
        assert "main" in push_branches, "push トリガーに main ブランチが含まれていません"
        assert "main" in pr_branches, "pull_request トリガーに main ブランチが含まれていません"
