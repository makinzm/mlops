# lefthook pre-commit フックと GitHub Actions CI の手動確認手順

## 前提条件

- [devenv](https://devenv.sh/getting-started/) がインストール済みであること
- `devenv shell` に入れること

---

## 1. devenv 環境のセットアップ

```bash
devenv shell
```

初回は Nix パッケージのダウンロードに数分かかります。以降はキャッシュが効くため高速になります。

---

## 2. lefthook フックのインストール

devenv shell 内で実行します。

```bash
lefthook install
```

成功すると `.git/hooks/pre-commit` が作成されます。

```
sync hooks: ✔️ (pre-commit)
```

---

## 3. pre-commit フックの手動実行

コミット前にすべてのチェックを手動で走らせる場合：

```bash
lefthook run pre-commit
```

期待出力（全チェック通過時）：

```
summary:
✔️  ruff-check
✔️  ruff-format
✔️  mypy
✔️  actionlint
```

---

## 4. 各チェックの個別実行

devenv shell 内で以下を実行します。

### ruff lint チェック

```bash
uv run --extra dev ruff check .
```

### ruff フォーマットチェック

```bash
uv run --extra dev ruff format --check .
```

### mypy 型チェック

```bash
uv run --extra dev mypy src/
```

### actionlint（GitHub Actions ワークフローの Lint）

```bash
actionlint .github/workflows/ci.yml
```

---

## 5. pytest の実行

```bash
uv run --extra dev pytest --cov=src --cov-report=xml
```

---

## 6. git commit による pre-commit フックの動作確認

```bash
git add <変更ファイル>
git commit -m "test commit"
```

すべてのチェックが通れば commit が作成されます。いずれかが失敗した場合は commit が中断され、エラーメッセージが表示されます。

---

## 7. GitHub Actions CI の確認

PR を作成するか、`main` ブランチにプッシュすると、GitHub Actions が自動的にトリガーされます。

以下の 4 ジョブが実行されます：

| ジョブ | 内容 |
|---|---|
| `lint` | ruff check / ruff format --check |
| `type-check` | mypy src/ |
| `test` | pytest --cov |
| `actionlint` | .github/workflows/*.yml の Lint |

CI の状態は GitHub リポジトリの **Actions** タブから確認できます。

---

## トラブルシューティング

### `lefthook: command not found`

devenv shell に入っていない可能性があります。

```bash
devenv shell
lefthook run pre-commit
```

### `devenv shell` が遅い

初回は Nix パッケージのビルドが発生します。2 回目以降はキャッシュが効きます。

### mypy が `types-PyYAML` を要求する

```bash
uv sync --extra dev
```

### CI で `devenv` が見つからない

`cachix/install-nix-action` と `cachix/cachix-action` のステップが正しく設定されているか確認してください（`ci.yml` 参照）。
