# Kaggle データダウンロード 手順書

---

## クイックスタート

### Titanic コンペデータをダウンロードする

```bash
uv run python -m src
```

デフォルト設定（`conf/downloader/kaggle.yaml`）が `mode: competition / competition: titanic` なので、
引数なしで Titanic データがダウンロードされる。

### 任意のデータセットをダウンロードする

```bash
uv run python -m src downloader.mode=dataset downloader.dataset=owner/dataset-name
```

### 任意のコンペデータをダウンロードする

```bash
uv run python -m src downloader.mode=competition downloader.competition=titanic
```

---

## 初回セットアップ

### 1. Kaggle API トークンの取得と保存

1. https://www.kaggle.com/settings を開く
2. "API" セクションの **"Create New Token"** をクリック
3. 表示されたトークン文字列をコピーする
4. 以下のコマンドで保存する:

```bash
mkdir -p ~/.kaggle
echo "your-token-here" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

> `~/.kaggle/access_token` はホームディレクトリに置くため、プロジェクトに関係なく使い回せる。

### 2. 依存パッケージのインストール

```bash
uv sync
```

---

## 識別子の調べ方

### データセットの場合

URL: `https://www.kaggle.com/datasets/owner/dataset-name`
→ 識別子: `owner/dataset-name`

例: `https://www.kaggle.com/datasets/titanic/titanic` → `titanic/titanic`

### コンペティションの場合

URL: `https://www.kaggle.com/competitions/competition-name`
→ 識別子: `competition-name`

例: `https://www.kaggle.com/competitions/titanic` → `titanic`

---

## 出力先

`output_dir`（デフォルト: `data/2026/Q1/raw/`）に保存される。
`data/` は `.gitignore` 済みのため、大容量ファイルが誤ってコミットされることはない。

出力先を変更する場合:

```bash
uv run python -m src output_dir=data/custom/path
```

---

## 設定ファイル

| ファイル | 内容 |
|---|---|
| `conf/downloader/kaggle.yaml` | mode / dataset / competition |
| `conf/usecase/download_dataset.yaml` | output_dir / unzip / force |

---

## よくあるエラー

### `Could not find kaggle.json` / 認証エラー

`~/.kaggle/access_token` にトークンが正しく保存されているか確認する。

```bash
cat ~/.kaggle/access_token    # トークン文字列が表示されること
ls -la ~/.kaggle/access_token # パーミッションが 600 であること
```

### `dataset が未指定です`

`mode=dataset` なのに `dataset` が未設定。識別子を指定する:

```bash
uv run python -m src downloader.mode=dataset downloader.dataset=owner/dataset-name
```

### `404 Not Found`

識別子が間違っている。Kaggle の URL から正確にコピーする。

### `403 Forbidden`

コンペの利用規約に未同意。Kaggle のコンペページで "I Understand and Accept" をクリックする。
