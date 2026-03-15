# Kaggle データダウンロード

## 実行

```bash
uv run python -m src usecase=download_dataset
```

デフォルトで Titanic データを `data/2026/Q1/raw/` にダウンロードする。

---

## 設定ファイル

| ファイル | 役割 |
|---------|------|
| `conf/usecase/download_dataset.yaml` | **どこに保存するか**（`output_dir`）と動作オプション |
| `conf/downloader/kaggle.yaml` | **何をダウンロードするか**（コンペ名・データセット名） |

```yaml
# conf/usecase/download_dataset.yaml
output_dir: "data/2026/Q1/raw"  # ダウンロード先
unzip: true
force: true
```

```yaml
# conf/downloader/kaggle.yaml
mode: "competition"             # competition / dataset
competition: "titanic"          # コンペ名
dataset: null                   # mode=dataset のとき: owner/dataset-name
```

---

## 初回セットアップ

```bash
# Kaggle API トークンを保存（https://www.kaggle.com/settings → Create New Token）
mkdir -p ~/.kaggle
echo "your-token-here" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token

uv sync
```

---

## 識別子の調べ方

- コンペ: `https://www.kaggle.com/competitions/titanic` → `titanic`
- データセット: `https://www.kaggle.com/datasets/owner/name` → `owner/name`
