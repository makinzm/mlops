# 0050: テンプレートワークフロー

このリポジトリをテンプレートとして Kaggle コンペ用リポジトリを作成し、
コンペ終了後に汎用的な改善をテンプレートに戻すまでのワークフロー。

---

## 前提

- テンプレートリポジトリ: `github.com/makinzm/mlops` (本リポジトリ)
- コンペ用リポジトリ: Private リポジトリとして別途作成
- `.env` や `~/.kaggle/access_token` は全リポジトリで共通

---

## 1. 共通設定の一元管理（初回のみ）

コンペ用リポジトリを作るたびに `.env` を書き直す手間を省くため、
マスターファイルを1箇所に置き、各リポジトリからシンボリックリンクする。

### 1.1 マスターディレクトリを作成

```bash
mkdir -p ~/.config/mlops
```

### 1.2 マスター `.env` を作成

```bash
cp /path/to/mlops/.env ~/.config/mlops/.env
```

内容例:

```env
# GCP 設定
GCP_PROJECT=mlops-titanic-123456
GCP_REGION=asia-northeast1

# Slack 通知（任意）
# SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../xxx
```

> **注意**: Kaggle 認証は `~/.kaggle/access_token` を使用するため `.env` には含めない。

### 1.3 既存のテンプレートリポジトリにリンクを貼る

```bash
cd /path/to/mlops
ln -sf ~/.config/mlops/.env .env
```

以降、すべてのリポジトリで同じ手順を行う（後述の「2.3」参照）。

---

## 2. コンペ用リポジトリの作成

### 2.1 テンプレートから clone

```bash
git clone git@github.com:makinzm/mlops.git kaggle-{competition_name}
cd kaggle-{competition_name}
```

### 2.2 remote を設定

`origin` をコンペ用の Private リポジトリに向け、テンプレートを `upstream` として残す。

```bash
# コンペ用リポジトリを GitHub で作成済みとする
git remote set-url origin git@github.com:{your_username}/kaggle-{competition_name}.git
git remote add upstream git@github.com:makinzm/mlops.git
```

確認:

```bash
git remote -v
# origin    git@github.com:{your_username}/kaggle-{competition_name}.git (fetch)
# origin    git@github.com:{your_username}/kaggle-{competition_name}.git (push)
# upstream  git@github.com:makinzm/mlops.git (fetch)
# upstream  git@github.com:makinzm/mlops.git (push)
```

### 2.3 共通設定をリンク

```bash
ln -sf ~/.config/mlops/.env .env
```

これだけで `.env` のセットアップは完了。GCP 設定等が即座に使える。

### 2.4 依存インストール & 動作確認

```bash
uv sync
uv run python -m src --help
```

### 2.5 テンプレートの更新を取り込む（任意）

コンペ期間中にテンプレートが更新された場合:

```bash
git fetch upstream
git merge upstream/main
```

---

## 3. コンペ終了後のバックポート

コンペで追加した汎用的な関数やユーティリティをテンプレートに戻す手順。

### 3.1 方針: どのコミットを戻すか判断する

バックポート対象の基準:

| 対象 | 例 |
|------|-----|
| 汎用的な前処理関数 | `polars:clip_outliers`, `sklearn:target_encode` 等 |
| 新しいモデルアーキテクチャ | CNN, Transformer 等のラッパー |
| インフラ改善 | Vertex AI の新しい実行パターン等 |
| バグ修正 | テンプレート由来のバグ |

バックポート **しない** もの:

| 対象外 | 理由 |
|--------|------|
| コンペ固有の conf ファイル | `conf/competition/{name}/` 以下 |
| コンペ固有のデータパス | 他のコンペでは使わない |
| 実験ログ・モデルファイル | データは持ち込まない |

### 3.2 コンペリポジトリでバックポート候補を特定

```bash
cd kaggle-{competition_name}

# テンプレートとの差分をファイル単位で確認
git diff upstream/main --stat

# src/ 以下の変更のみ確認（conf/ やデータは除外）
git diff upstream/main -- src/ tests/
```

### 3.3 テンプレートにコンペリポジトリを remote 追加

```bash
cd /path/to/mlops  # テンプレートリポジトリ
git fetch upstream  # 最新化

# コンペリポジトリを一時的に remote 追加
git remote add competition git@github.com:{your_username}/kaggle-{competition_name}.git
git fetch competition
```

### 3.4 cherry-pick で取り込む

```bash
# バックポート用ブランチを作成
git checkout -b backport/{competition_name} origin/main

# 対象コミットを cherry-pick（コミットハッシュはコンペリポジトリの git log で確認）
git cherry-pick {commit_hash_1}
git cherry-pick {commit_hash_2}
```

> **ヒント**: コンペ中に「これはテンプレートにも欲しい」と思った変更は、
> 独立したコミットにしておくと cherry-pick しやすい。
> コミットメッセージに `[backport]` タグをつけておくと後で探しやすい。

```bash
# コンペリポジトリ側で [backport] タグ付きコミットを探す
git log --oneline competition/main | grep '\[backport\]'
```

### 3.5 コンフリクト解消

cherry-pick でコンフリクトが発生した場合:

```bash
# コンフリクトファイルを確認
git status

# 手動で解消後
git add {resolved_files}
git cherry-pick --continue
```

### 3.6 テスト実行 & PR 作成

```bash
# テストが通ることを確認
uv run pytest tests/

# push して PR 作成
git push -u origin backport/{competition_name}
gh pr create \
  --title "backport: {competition_name} から汎用関数を移植" \
  --body "## Summary
- {competition_name} コンペで追加した汎用関数をテンプレートに反映

## cherry-pick 元コミット
- {commit_hash_1}: {description}
- {commit_hash_2}: {description}

## Test plan
- [ ] \`uv run pytest tests/\` が通る
- [ ] 既存のコンペ conf で \`uv run python -m src --help\` が動作する"
```

### 3.7 クリーンアップ

PR がマージされたら一時 remote を削除:

```bash
git remote remove competition
```

---

## 4. ワークフロー全体図

```
┌─────────────────────────────────────────────────────┐
│                  テンプレートリポジトリ                  │
│                  (makinzm/mlops)                      │
│                                                       │
│  upstream ◄──────────────────────── cherry-pick PR    │
└───────┬───────────────────────────────────▲───────────┘
        │ clone                             │
        ▼                                   │
┌─────────────────────────────────┐         │
│    コンペ用リポジトリ (Private)    │         │
│                                  │ ────────┘
│  origin: kaggle-{competition}    │  バックポート
│  upstream: makinzm/mlops         │  ([backport] コミット)
│  .env → ~/.config/mlops/.env     │
└─────────────────────────────────┘
```

---

## 5. チェックリスト

### コンペ開始時

- [ ] `git clone` でテンプレートから作成
- [ ] `origin` をコンペ用 Private リポジトリに変更
- [ ] `upstream` にテンプレートを追加
- [ ] `ln -sf ~/.config/mlops/.env .env` でリンク
- [ ] `uv sync` で依存インストール

### コンペ期間中

- [ ] テンプレートに戻したい変更は独立コミット + `[backport]` タグ
- [ ] 必要に応じて `git fetch upstream && git merge upstream/main`

### コンペ終了後

- [ ] `git diff upstream/main -- src/ tests/` でバックポート候補を特定
- [ ] テンプレートで `backport/{competition_name}` ブランチを作成
- [ ] `cherry-pick` で対象コミットを取り込み
- [ ] テスト通過を確認して PR 作成
- [ ] 一時 remote を削除
