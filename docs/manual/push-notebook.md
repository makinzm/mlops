# Kaggle Notebook のプッシュ・実行手順

ローカルで開発したパイプラインを Kaggle Notebook として生成・プッシュし、実行・提出するまでの手順。

---

## 前提条件

### 1. Kaggle 認証トークンの取得

1. [https://www.kaggle.com/settings](https://www.kaggle.com/settings) を開く
2. 「API」セクションの「Create New Token」をクリック
3. ダウンロードされた `kaggle.json` の `key` の値をコピーする
4. トークンを保存する:

```bash
echo "ここにkeyの値を貼り付け" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

### 2. `.env` の設定

```bash
cp .env.example .env
# KAGGLE_USERNAME=your_kaggle_username を記入
```

---

## 基本ワークフロー

### ステップ 1: src/ と conf/ を Kaggle Dataset にアップロード

```bash
# 初回（Dataset 新規作成）
uv run python -m src usecase=create_source_dataset

# コード・設定変更後（バージョン更新）
uv run python -m src usecase=update_source_dataset source_dataset.version_message="add target encoding"
```

アップロードされる内容:

```
/kaggle/input/mlops-pipeline-src/
  domain/, infrastructure/, usecase/, ...   ← src/ の中身
  conf/                                     ← Hydra 設定ファイル
  requirements.txt                          ← uv export で自動生成
```

### ステップ 2: Notebook を生成して Kaggle にプッシュ

```bash
uv run python -m src usecase=push_notebook notebook.competition=titanic
```

### ステップ 3: Kaggle Notebook で実行

1. ブラウザで `https://www.kaggle.com/code/{username}/titanic-pipeline` を開く
2. セル 2 の `COMPETITION_SLUG` と `RECIPE` を確認する
3. 「Run All」をクリック
4. `/kaggle/working/inference/.../submission.csv` が生成されたら「Submit to Competition」

---

## Notebook の構成

生成される `notebook.ipynb` は 3 セル固定。**編集するのはセル 2 だけ。**

### セル 1: 環境セットアップ（編集不要）

- 未インストールのパッケージだけを `requirements.txt` から pip install する（スマートインストール）
- `/kaggle/working/src → /kaggle/input/mlops-pipeline-src` のシンボリックリンクを作成し `from src.xxx import ...` を有効化する

スマートインストールの動作:

| 状況 | 動作 |
|------|------|
| 未インストール | `pip install` する |
| インストール済み・バージョン一致 | スキップ |
| インストール済み・メジャーバージョン違い | `WARNING: pkg installed=X required=Y (major version differs, skipping)` を出してスキップ |

### セル 2: 設定（ここだけ編集する）

```python
COMPETITION_SLUG = "titanic"
RECIPE = "base"   # conf/recipe/ 以下のファイル名

# Notebook 上での設定上書き（不要なら空のまま）
# 例: {"lgbm.num_leaves": 64, "preprocess.cv": False}
OVERRIDES: dict = {}
```

### セル 3: パイプライン実行（編集不要）

```python
from src.infrastructure.kaggle.notebook_runner import NotebookPipelineRunner

runner = NotebookPipelineRunner(
    competition_slug=COMPETITION_SLUG,
    recipe=RECIPE,
    overrides=OVERRIDES,
)
submission_path = runner.run()
print(f"[INFO] submission: {submission_path}")
```

---

## 部分実行（Inference だけ実行したいとき）

全パイプラインではなく Inference だけ実行したい場合は、**inference のみの recipe** を作成する。

### 1. recipe ファイルを作成

```yaml
# conf/recipe/inference_only.yaml
steps:
  - usecase: inference
    job_id: titanic_inference
    # test_path と output_dir は NotebookPipelineRunner が Kaggle パスで自動上書きする
    test_path: PLACEHOLDER
    output_dir: PLACEHOLDER
    models:
      - job_id: titanic_lgbm
        model_path: PLACEHOLDER
```

> **注意**: `test_path` / `output_dir` / `models[].model_path` は
> `NotebookPipelineRunner` が `/kaggle/working/` 以下のパスで自動的に差し替えるため
> yaml 側は `PLACEHOLDER` で問題ない。

### 2. Dataset を更新してプッシュ

```bash
uv run python -m src usecase=update_source_dataset source_dataset.version_message="add inference_only recipe"
uv run python -m src usecase=push_notebook notebook.competition=titanic
```

### 3. Notebook のセル 2 を変更

```python
COMPETITION_SLUG = "titanic"
RECIPE = "inference_only"   # ← 変更
OVERRIDES: dict = {}
```

### OVERRIDES でパスを手動指定する場合

特定のモデルを使いたい場合や test_path を変えたい場合は OVERRIDES で上書きできる:

```python
OVERRIDES = {
    # 前処理済みデータを別の場所から読む
    "test_path": "/kaggle/input/my-preprocessed-data/test.parquet",
    # 別の job_id のモデルを使う
    "models.0.job_id": "titanic_lgbm_v2",
}
```

---

## インターネット無効コンペへの対応

コンペによっては Notebook のインターネット接続が禁止されている。
その場合は依存パッケージを別の Kaggle Dataset として事前アップロードしておく。

### 1. パッケージをローカルにダウンロード

```bash
# .packages/ にダウンロード（.gitignore 対象）
pip download -r requirements.txt -d .packages/ --platform manylinux2014_x86_64 --only-binary=:all:
```

> `--platform manylinux2014_x86_64 --only-binary=:all:` を指定することで
> Kaggle (Linux x86_64) 向けのバイナリ wheel をダウンロードする。

### 2. パッケージ Dataset を Kaggle に登録

`src/packages/` ディレクトリを作成して `dataset-metadata.json` を用意する:

```json
{
  "title": "mlops-packages",
  "id": "{your_username}/mlops-packages",
  "licenses": [{"name": "CC0-1.0"}]
}
```

```bash
# 手動で kaggle コマンドを使ってアップロード
uv run kaggle datasets create -p .packages/ -r zip
```

以降の更新:

```bash
uv run kaggle datasets version -p .packages/ -m "update packages" -r zip
```

### 3. Notebook の dataset_sources に追加

`conf/usecase/push_notebook.yaml` を編集して `packages_dataset` を追加:

```yaml
notebook:
  competition: titanic
  kernel_slug: titanic-pipeline
  src_dataset: mlops-pipeline-src
  packages_dataset: mlops-packages   # ← 追加
  enable_gpu: false
  enable_internet: false             # ← false に変更
```

> **現状の制限**: `packages_dataset` と `enable_internet: false` は
> `conf/usecase/push_notebook.yaml` と `kernel-metadata.json` の手動編集が必要。
> 将来的には `push_notebook` usecase が自動反映する予定。

手動で `notebooks/titanic/kernel-metadata.json` を編集する場合:

```json
{
  "enable_internet": false,
  "dataset_sources": [
    "your_username/mlops-pipeline-src",
    "your_username/mlops-packages"
  ]
}
```

その後 Notebook をプッシュ:

```bash
uv run kaggle kernels push -p notebooks/titanic/
```

### 4. Notebook セル 2 でオフラインインストールを指定

セル 2 の `OVERRIDES` にパッケージディレクトリを追加:

```python
COMPETITION_SLUG = "titanic"
RECIPE = "base"
OVERRIDES: dict = {}

# インターネット無効環境用: パッケージを Dataset から読み込む
PACKAGES_DIR = "/kaggle/input/mlops-packages"
```

セル 1 の `_smart_install` 呼び出しを以下に変更:

```python
# オフラインインストール（インターネット無効コンペ用）
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q',
    '--no-index', f'--find-links={PACKAGES_DIR}',
    '-r', f'/kaggle/input/mlops-pipeline-src/requirements.txt'
])
```

> セル 1 は「編集不要」だが、インターネット無効コンペのみここを変更する必要がある。
> 将来的には `PACKAGES_DIR` を指定するだけで自動切り替えする予定。

---

## パス対応表

| 用途 | ローカル | Kaggle Notebook |
|------|---------|----------------|
| raw データ（input） | `data/2026/Q1/raw/titanic/` | `/kaggle/input/titanic/` |
| 前処理済みデータ | `data/2026/Q1/processed/` | `/kaggle/working/processed/` |
| モデル | `models/titanic/` | `/kaggle/working/models/` |
| 推論結果・submission | `data/2026/Q1/inference/` | `/kaggle/working/inference/` |

---

## 設定変更リファレンス

### コンペを変える

```bash
uv run python -m src usecase=push_notebook notebook.competition=otto
```

### kernel_slug（URL名）を変える

```bash
uv run python -m src usecase=push_notebook notebook.competition=titanic notebook.kernel_slug=titanic-v2
```

### GPU を有効にする

```bash
uv run python -m src usecase=push_notebook notebook.competition=titanic notebook.enable_gpu=true
```

### src_dataset のスラッグを変える

```yaml
# conf/usecase/push_notebook.yaml
notebook:
  src_dataset: my-custom-src-dataset
```

---

## トラブルシューティング

### `RuntimeError: Kaggle 認証に失敗しました`

- `~/.kaggle/access_token` が存在するか確認する
- トークンに改行が含まれていないか確認する: `cat -A ~/.kaggle/access_token`

### `id: None/titanic-pipeline` になる

- `.env` の `KAGGLE_USERNAME` が設定されているか確認する

### `ModuleNotFoundError: No module named 'src'`

- セル 1 を実行したか確認する（「Run All」で全セル実行すること）
- Dataset として `mlops-pipeline-src` が Notebook に追加されているか確認する
  (`kernel-metadata.json` の `dataset_sources` を確認)

### `FileNotFoundError: recipe yaml が見つかりません`

- `RECIPE` に指定したファイルが `conf/recipe/` に存在するか確認する
- `update_source_dataset` を実行して最新 conf/ をアップロードしたか確認する
