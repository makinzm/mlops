# lefthook pre-commit + GitHub Actions CI の自動化

## 背景・目的

`devenv.nix` に `lefthook` と `actionlint` が追加されたが、実際のフック設定（`lefthook.yml`）と GitHub Actions ワークフロー（`.github/workflows/ci.yml`）がまだ存在しない。`pyproject.toml` には `ruff`/`m-y-p-y`/`pytest` が設定済みのため、これらを自動実行する仕組みを整える。

## 対象ファイル

| ファイル | 説明 |
|---|---|
| `lefthook.yml` | pre-commit フック設定 |
| `.github/workflows/ci.yml` | GitHub Actions CI ワークフロー |
| `tests/test_ci_setup.py` | 設定ファイルの妥当性検証テスト |
| `docs/manual/lefthook-ci.md` | 手動確認手順ドキュメント |

## 実装方針

### lefthook.yml

`pre-commit` フックで以下を並列実行する：

- `ruff check`：Lint チェック
- `ruff format --check`：フォーマットチェック
- `m-y-p-y`：型チェック
- `actionlint`：GitHub Actions ワークフローの Lint

### .github/workflows/ci.yml

PR および `main` へのプッシュ時に以下のジョブを実行する：

- `lint`：ruff check / ruff format チェック
- `type-check`：m-y-p-y による型チェック
- `test`：pytest + カバレッジレポート
- `actionlint`：ワークフローファイルの Lint

## テスト戦略

`tests/test_ci_setup.py` で以下を検証する：

1. `lefthook.yml` が YAML としてパース可能か
2. 必須コマンド（`ruff-check`, `ruff-format`, `m-y-p-y`, `actionlint`）が定義されているか
3. `.github/workflows/ci.yml` が YAML としてパース可能か
4. 必須ジョブ（`lint`, `type-check`, `test`, `actionlint`）が定義されているか

## DoD チェックリスト

- [ ] Red → Green → Refactor のサイクルが Commit 単位で行われていること
- [ ] テストが先に書かれており、なぜそのテストが必要か説明されていること
- [ ] 手動確認手順が `docs/manual/lefthook-ci.md` に記載されていること
- [ ] lefthook と GitHub Actions で自動実行されること
- [ ] `lefthook run pre-commit` でフックが動作すること
- [ ] `actionlint .github/workflows/ci.yml` でワークフローが有効と確認されること
