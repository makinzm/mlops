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

### 3.1 方針: どのファイルを戻すか判断する

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

### 3.2 テンプレートにコンペリポジトリを remote 追加

```bash
cd /path/to/mlops  # テンプレートリポジトリ

# コンペリポジトリを一時的に remote 追加
git remote add competition git@github.com:{your_username}/kaggle-{competition_name}.git
git fetch competition
```

### 3.3 差分をファイル単位で確認

```bash
# コンペリポジトリで変更されたファイル一覧を確認
git diff main...competition/main --stat

# src/ と tests/ に絞って確認（conf/ やデータは除外）
git diff main...competition/main --stat -- src/ tests/

# 特定ディレクトリの中身を詳しく見たい場合
git diff main...competition/main -- src/domain/preprocessing/
```

### 3.4 ファイルパス指定で一括取り込み

`git checkout` でコンペリポジトリからファイル単位・ディレクトリ単位で取り込む。
コミット履歴ではなくファイルの最終状態を持ってくるので、コミットの粒度を気にしなくてよい。

```bash
# バックポート用ブランチを作成
git checkout -b backport/{competition_name} origin/main

# ファイル単位で取り込み
git checkout competition/main -- src/domain/preprocessing/clip_outliers.py
git checkout competition/main -- tests/domain/preprocessing/test_clip_outliers.py

# ディレクトリ単位で一括取り込み
git checkout competition/main -- src/domain/model/transformer/
git checkout competition/main -- tests/domain/model/transformer/
```

この時点で取り込んだファイルはすべて staging 済みの状態になる。

### 3.5 staging を解除して整理コミットを作成

`git reset --soft` の考え方と同じく、一度 staging を解除してから
関心事ごとにコミットを分ける。

```bash
# staging を解除（ファイルは working tree に残る）
git restore --staged .

# 変更内容を確認
git status
git diff

# 関心事ごとに add & commit
git add src/domain/preprocessing/clip_outliers.py tests/domain/preprocessing/test_clip_outliers.py
git commit -m "feat: clip_outliers 前処理関数を追加"

git add src/domain/model/transformer/ tests/domain/model/transformer/
git commit -m "feat: Transformer モデルラッパーを追加"
```

> **ヒント**: 取り込んだファイルにコンペ固有のハードコードがないか `git diff` で確認し、
> 汎用的な形に修正してからコミットする。

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

## 取り込んだファイル
- src/domain/preprocessing/clip_outliers.py
- src/domain/model/transformer/
- （対応するテストファイル）

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
│  upstream ◄─────────────────── backport PR             │
└───────┬───────────────────────────────────▲───────────┘
        │ clone                             │
        ▼                                   │
┌─────────────────────────────────┐         │
│    コンペ用リポジトリ (Private)    │         │
│                                  │ ────────┘
│  origin: kaggle-{competition}    │  git checkout で
│  upstream: makinzm/mlops         │  ファイル単位取り込み
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

- [ ] 必要に応じて `git fetch upstream && git merge upstream/main`

### コンペ終了後

- [ ] テンプレートに competition remote を追加
- [ ] `git diff main...competition/main --stat -- src/ tests/` でバックポート候補を特定
- [ ] `git checkout competition/main -- {paths}` でファイル単位取り込み
- [ ] `git restore --staged .` → 関心事ごとにコミット整理
- [ ] テスト通過を確認して PR 作成
- [ ] 一時 remote を削除
