# Kaggle データダウンロード

## 実行

```bash
uv run python -m src +usecase=download_dataset
```

デフォルトで Titanic データを `data/2026/Q1/raw/titanic/` にダウンロードする。

---

## 設定ファイル

`conf/usecase/download_dataset.yaml` に全ての設定が集約されている。

```yaml
# conf/usecase/download_dataset.yaml
output_dir: "data/2026/Q1/raw/${competition.name}"  # ダウンロード先（competition.name を補間）
unzip: true
force: true
source: kaggle
kaggle:
  mode: "competition"             # competition / dataset
  competition: ${competition.name}  # conf/competition/{name}/competition.yaml の name を参照
  dataset: null                   # mode=dataset のとき: owner/dataset-name
```

コンペは `conf/competition/titanic/competition.yaml` の `name` を参照する。

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
