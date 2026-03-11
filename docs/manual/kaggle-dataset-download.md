# Kaggle データセットダウンロード 手動手順書

## 前提条件

- [Kaggle アカウント](https://www.kaggle.com)を持っていること
- `uv` がインストールされていること

---

## 1. 初回セットアップ

### 1-1. Kaggle API トークンの取得と保存

1. https://www.kaggle.com/settings を開く
2. "API" セクションの **"Create New Token"** をクリック
3. 表示されたトークン文字列をコピーする
4. 以下のコマンドでトークンを保存する:

```bash
mkdir -p ~/.kaggle
echo "your-token-here" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

> `~/.kaggle/access_token` はホームディレクトリに置くため、プロジェクトに関係なく使い回せる。

### 1-2. 依存パッケージのインストール

```bash
uv sync
```

---

## 2. データセット識別子の調べ方

Kaggle のデータセット識別子は `owner/dataset-name` の形式。

### データセットの場合

1. ダウンロードしたいデータセットのページを Kaggle で開く
2. URL が `https://www.kaggle.com/datasets/owner/dataset-name` の形式
3. `owner/dataset-name` の部分が識別子

例: `https://www.kaggle.com/datasets/titanic/titanic` → 識別子は `titanic/titanic`

### コンペティションの場合

1. コンペのページを開く
2. URL が `https://www.kaggle.com/competitions/competition-name` の形式
3. `competition-name` の部分が識別子

例: `https://www.kaggle.com/competitions/titanic` → 識別子は `titanic`

---

## 3. データセットのダウンロード

### 3-1. 設定ファイルの確認・編集

`conf/downloader/kaggle.yaml` でダウンロード対象を指定する:

```yaml
type: "kaggle"
mode: "dataset"                 # "dataset" or "competition"
dataset: "owner/dataset-name"   # ← 手順 2 で調べた識別子
competition: null
```

`conf/usecase/download_dataset.yaml` で出力先などを指定する:

```yaml
seed: 42
output_dir: "data/2026/Q1/raw"  # ← ダウンロード先（プロジェクトルートからの相対パス）
unzip: true
force: false
```

### 3-2. 実行

```bash
uv run python -m src usecase=download_dataset
```

設定をコマンドラインで上書きすることも可能:

```bash
# データセットを直接指定
uv run python -m src usecase=download_dataset downloader.dataset=titanic/titanic

# コンペデータをダウンロード
uv run python -m src usecase=download_dataset downloader.mode=competition downloader.competition=titanic
```

### 3-3. 実行結果の確認

```
Downloaded to: data/2026/Q1/raw
Files: 3
Commit: a1b2c3d...
```

- ダウンロードされたファイルは **`output_dir`（デフォルト: `data/2026/Q1/raw/`）** に保存される
- `data/` は `.gitignore` 済みのため、大容量ファイルが誤ってコミットされることはない
- `commit_hash` により、どのバージョンのコードでダウンロードしたかが記録される（再現性の担保）

---

## 4. よくあるエラー

### `Could not find kaggle.json` / 認証エラー

`~/.kaggle/access_token` にトークンが正しく保存されているか確認する。

```bash
cat ~/.kaggle/access_token   # トークン文字列が表示されること
ls -la ~/.kaggle/access_token  # パーミッションが 600 であること
```

設定し直す場合:

```bash
echo "your-token-here" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

### `404 Not Found`

データセット識別子が間違っている。手順 2 で URL から正確にコピーする。

### `403 Forbidden`

コンペの利用規約に同意していない。Kaggle のコンペページで "I Understand and Accept" をクリックする。
