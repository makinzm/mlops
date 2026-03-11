# Kaggle データセットダウンロード 手動手順書

## 前提条件

- [Kaggle アカウント](https://www.kaggle.com)を持っていること
- `uv` がインストールされていること

---

## 1. 初回セットアップ

### 1-1. Kaggle API トークンの取得

1. https://www.kaggle.com/settings を開く
2. "API" セクションの **"Create New Token"** をクリック
3. 表示されたトークン文字列をコピーする

### 1-2. 認証情報の設定

`.env.example` をコピーしてトークンを設定する:

```bash
cp .env.example .env
```

`.env` を編集:

```
KAGGLE_API_TOKEN=your-token-here
```

> `.env` は `.gitignore` 済みのため、誤ってコミットされることはない。

### 1-3. 依存パッケージのインストール

```bash
uv sync --group dev --group kaggle
```

---

## 2. データセットのダウンロード

### 2-1. 設定ファイルの確認・編集

`conf/usecase/download_dataset.yaml` を用途に合わせて編集する:

```yaml
seed: 42
data_from: "kaggle"
output_dir: "data/2026/Q1/raw"   # ← ダウンロード先
unzip: true
force: false

kaggle:
  mode: "dataset"                 # "dataset" or "competition"
  dataset: "owner/dataset-name"   # ← Kaggle のデータセット識別子
  competition: null
```

### 2-2. 実行

```bash
uv run python -m src usecase=download_dataset
```

設定をコマンドラインで上書きすることも可能:

```bash
# データセットを直接指定
uv run python -m src usecase=download_dataset kaggle.dataset=titanic/titanic

# コンペデータをダウンロード
uv run python -m src usecase=download_dataset kaggle.mode=competition kaggle.competition=titanic
```

### 2-3. 実行結果の確認

```
Downloaded to: data/2026/Q1/raw
Files: 3
Commit: a1b2c3d...
```

ダウンロードされたファイルは `output_dir` に保存される。`commit_hash` により、どのバージョンのコードでダウンロードしたかが記録される。

---

## 3. よくあるエラー

### `Could not find kaggle.json` / `SystemExit: 1`

認証情報が設定されていない。`.env` に `KAGGLE_API_TOKEN` が設定されているか確認する。

### `404 Not Found`

データセット識別子が間違っている。Kaggle のデータセットページ URL（`kaggle.com/datasets/owner/name`）から `owner/name` を確認する。

### `403 Forbidden`

コンペの利用規約に同意していない。Kaggle のコンペページで "I Understand and Accept" をクリックする。
