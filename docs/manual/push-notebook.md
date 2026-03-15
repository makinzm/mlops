# Kaggle Notebook のプッシュ手順

このドキュメントでは、ローカルで開発したパイプラインを Kaggle Notebook として生成・プッシュする手順を説明する。

---

## 前提条件

### 1. Kaggle 認証トークンの取得

1. [https://www.kaggle.com/settings](https://www.kaggle.com/settings) を開く
2. 「API」セクションの「Create New Token」をクリック
3. ダウンロードされた `kaggle.json` を開き、`key` の値をコピーする
4. トークンをファイルに保存する:

```bash
echo "ここにkeyの値を貼り付け" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

### 2. `.env` の設定

```bash
cp .env.example .env
```

`.env` を開いて `KAGGLE_USERNAME` を記入する:

```
KAGGLE_USERNAME=your_kaggle_username
```

### 3. src/ コードを Kaggle Dataset として登録（初回のみ）

Notebook 実行時に Kaggle 環境から `src/` のコードを参照するため、
コードを「Kaggle Dataset（コードリポジトリ）」としてアップロードする。

#### Dataset のスラッグ名の確認方法

設定ファイル（`conf/usecase/create_source_dataset.yaml`）の `source_dataset.dataset_slug` に記載されたスラッグを使う。
デフォルトは `mlops-pipeline-src`。

Kaggle Dataset の URL は `https://www.kaggle.com/datasets/{username}/{dataset_slug}` になる。

#### Dataset の新規作成（初回のみ）

```bash
uv run python -m src usecase=create_source_dataset
```

実行すると `.staging/source_dataset_{YYYYMMDD_HHMMSS}/` にステージングディレクトリが作成され、
Kaggle に新しい Dataset が作成される。成功後にステージングディレクトリは自動削除される。

#### Dataset のバージョン更新（2回目以降）

```bash
uv run python -m src usecase=update_source_dataset source_dataset.version_message="update source code"
```

コードを変更するたびに実行してバージョンを更新する。
`version_message` は変更内容の説明を自由に記入する。

#### .kaggleignore によるファイル除外

プロジェクトルートの `.kaggleignore` に記載されたパターンのファイルは除外される。
デフォルトで以下が除外済み:

```
__pycache__/
*.pyc
.venv/
*.egg-info/
.pytest_cache/
```

#### 設定ファイルのカスタマイズ

`conf/usecase/create_source_dataset.yaml` を編集することで以下を変更できる:

| キー | デフォルト値 | 説明 |
|------|-------------|------|
| `source_dataset.src_dir` | `src` | アップロードするディレクトリ |
| `source_dataset.dataset_slug` | `mlops-pipeline-src` | Dataset のスラッグ（URL に使われる） |
| `source_dataset.title` | `mlops-pipeline-src` | Dataset のタイトル |
| `source_dataset.license_name` | `CC0-1.0` | ライセンス |
| `source_dataset.kaggleignore` | `.kaggleignore` | 除外パターンファイルのパス |

---

## Notebook の生成とプッシュ

### 基本コマンド

```bash
uv run python -m src usecase=push_notebook notebook.competition=titanic
```

### 出力ファイルの確認

```bash
ls outputs/push_notebook/titanic/
# .gitignore
# kernel-metadata.json
# notebook.ipynb
# README.md
```

### `kernel-metadata.json` の内容確認

```bash
cat outputs/push_notebook/titanic/kernel-metadata.json
```

```json
{
  "id": "your_username/titanic-pipeline",
  "title": "titanic-pipeline",
  "code_file": "notebook.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": ["your_username/mlops-pipeline-src"],
  "competition_sources": ["titanic"],
  "kernel_sources": []
}
```

---

## Notebook の構成

生成される `notebook.ipynb` は3セルで構成される。

### セル 1: 環境セットアップ（編集不要）

```python
import sys

# Notebook 環境でのパッケージインストール
!pip install -q polars lightgbm hydra-core omegaconf pyyaml python-dotenv

sys.path.insert(0, "/kaggle/input/mlops-pipeline-src")
```

### セル 2: 設定上書き（competition slug を変える場合のみ編集）

```python
# セル 2: 設定上書き（competition slug やパスを変える場合はここを編集する）
# KaggleEnvironment が KAGGLE_KERNEL_RUN_TYPE 環境変数を検出して
# /kaggle/input / /kaggle/working に自動解決する。
COMPETITION_SLUG = "titanic"
# input:  /kaggle/input/{COMPETITION_SLUG}/
# output: /kaggle/working/
```

### セル 3: パイプライン実行（編集不要）

`KaggleEnvironment` がパスを解決し、前処理 → 学習 → 推論を順次実行する。
推論結果は `/kaggle/working/inference/.../submission.csv` に出力される。

---

## パス対応表

Kaggle Notebook 上でのパスは以下のように自動解決される。

| 用途 | ローカル（参考） | Kaggle Notebook |
|------|----------------|----------------|
| raw データ（input） | `data/2026/Q1/raw/titanic/` | `/kaggle/input/titanic/` |
| 前処理済みデータ | `data/2026/Q1/processed/` | `/kaggle/working/processed/` |
| モデル | `models/titanic/` | `/kaggle/working/models/` |
| 推論結果・submission | `data/2026/Q1/inference/` | `/kaggle/working/inference/` |

---

## Kaggle 上での Notebook 実行と submission 提出

1. ブラウザで以下の URL を開く:
   ```
   https://www.kaggle.com/code/{your_username}/titanic-pipeline
   ```

2. 「Edit」→「Run All」をクリックして全セルを実行する

3. 実行完了後、「Output」タブで `/kaggle/working/inference/.../submission.csv` が生成されていることを確認する

4. 「Submit to Competition」ボタンをクリックして提出する

---

## 設定の変更方法

### competition を変える場合

```bash
uv run python -m src usecase=push_notebook notebook.competition=house-prices-advanced-regression-techniques
```

### kernel_slug（URL に使われる名前）を変える場合

```bash
uv run python -m src usecase=push_notebook notebook.competition=titanic notebook.kernel_slug=my-titanic-v2
```

### GPU を有効にする場合

```bash
uv run python -m src usecase=push_notebook notebook.competition=titanic notebook.enable_gpu=true
```

---

## トラブルシューティング

### `RuntimeError: Kaggle 認証に失敗しました`

- `~/.kaggle/access_token` が存在するか確認する
- トークンが正しいか確認する（改行が含まれていないこと）

### `kernel-metadata.json` の `id` が空になる

- `.env` の `KAGGLE_USERNAME` が設定されているか確認する
- `uv run python -m src` 実行前に `source .env` または `dotenv` が有効になっているか確認する

### Notebook の実行が失敗する

- セル2の `COMPETITION_SLUG` が Kaggle Dataset のスラッグと一致しているか確認する
- `kernel-metadata.json` の `dataset_sources` に `{username}/mlops-pipeline-src` が含まれているか確認する
