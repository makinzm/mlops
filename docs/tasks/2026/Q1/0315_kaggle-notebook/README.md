# Kaggle Notebook Push ワークフロー実装計画

- 作成日: 2026-03-15
- ブランチ: `feat/kaggle-notebook-push`

---

## 背景・目的

現在このプロジェクトでは、ローカル環境でデータ前処理・学習・推論をパイプラインとして実行できる。
しかし Kaggle コンペへの提出には「Kaggle Notebook 上でパイプラインを動かす」ステップが手作業になっており、以下の課題がある。

1. **パス問題**: ローカルは `data/2026/Q1/raw/titanic/` だが、Kaggle Notebook は input が `/kaggle/input/{slug}/`・output が `/kaggle/working/` 固定
2. **Notebook 生成が手作業**: セル構成・設定値のコピペが毎回必要
3. **プッシュが手作業**: Kaggle の UI から手動アップロードが必要

これらを解消し、**ワンコマンドで Notebook 生成 → Kaggle プッシュ** できる状態にする。

---

## 出来物のイメージ（ユーザー操作フロー）

### Step 1: 事前設定（初回のみ）

```bash
# ~/.kaggle/access_token にトークン文字列を保存
echo "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token

# .env を編集
cp .env.example .env
# KAGGLE_USERNAME=your_username を記入
```

### Step 2: Notebook 生成 + Kaggle プッシュ（メインコマンド）

```bash
uv run python -m src usecase=push_notebook notebook.competition=titanic

# 出力:
# [INFO] Notebook generated: outputs/push_notebook/titanic/notebook.ipynb
# [INFO] kernel-metadata.json generated
# [INFO] Pushing to Kaggle... (kernels push)
# [INFO] Done: https://www.kaggle.com/code/{username}/titanic-pipeline
```

### Step 3: 生成された Notebook の構造

Notebook は**最大3セル**で構成される。編集が必要なのはセル2だけ。

```
セル 1 — セットアップ（編集不要）
  !pip install -q polars lightgbm hydra-core omegaconf pyyaml
  import sys
  sys.path.insert(0, "/kaggle/input/mlops-pipeline-src")

セル 2 — 設定上書き（ここだけ編集すればパスが変わる）
  # KaggleEnvironment がデフォルトで /kaggle/input・/kaggle/working を使うが、
  # スラッグやパスを変えたい場合はここで上書きする。
  COMPETITION_SLUG = "titanic"          # /kaggle/input/{slug}/
  OUTPUT_BASE      = "/kaggle/working"  # /kaggle/working/

セル 3 — パイプライン実行（編集不要）
  from src.infrastructure.kaggle.environment import KaggleEnvironment
  from src.usecase.pipeline.pipeline import PipelineUseCase
  # KaggleEnvironment がパス解決を担い、PipelineUseCase を実行する
  # → 前処理・学習・推論が順に走り /kaggle/working/inference/.../submission.csv が生成される
```

### Step 4: Kaggle 上でブラウザ確認 → 実行 → 提出

```
1. https://www.kaggle.com/code/{username}/titanic-pipeline を開く
2. "Run All" をクリック
3. /kaggle/working/inference/.../submission.csv が生成されることを確認
4. "Submit to Competition" で提出
```

---

## パス解決の設計（重要）

Kaggle Notebook では input と output のルートが固定されており、
パイプライン全体の設定（前処理・学習・推論）に埋め込まれたパスをすべて書き換える必要がある。

### ローカル vs Kaggle のパス対応表

| 用途 | ローカル | Kaggle Notebook |
|------|---------|----------------|
| raw データ（input） | `data/2026/Q1/raw/titanic/` | `/kaggle/input/titanic/` |
| 前処理済みデータ（output） | `data/2026/Q1/processed/` | `/kaggle/working/processed/` |
| モデル（output） | `models/titanic/` | `/kaggle/working/models/` |
| 推論結果・submission（output） | `data/2026/Q1/inference/titanic/` | `/kaggle/working/inference/` |

### 解決方針

`KaggleEnvironment` は2種類のルートを解決する:

- `resolve_input_root(slug)` → Kaggle なら `/kaggle/input/{slug}`、ローカルならそのまま
- `resolve_output_root()` → Kaggle なら `/kaggle/working`、ローカルならそのまま

パイプライン設定ファイル（`preprocess/base.yaml`, `training/lgbm.yaml`, `inference/titanic_ensemble.yaml`）の
`inputs[].path`, `output_dir`, `test_path`, `models` に埋め込まれたローカルパスを、
Notebook セル3で `KaggleEnvironment` を通して解決した設定に差し替えてから実行する。

Hydra Config は直接書き換えず、`OmegaConf.to_container` で plain dict に変換後に
パスだけ上書きして `DictConfig` を再構築する（caution.md 記載の struct モード回避パターン）。

---

## 手動実行手順の概要（`docs/manual/push-notebook.md` ドラフト）

### 前提条件

- `~/.kaggle/access_token` が存在する
- `.env` に `KAGGLE_USERNAME` が設定されている
- Kaggle 上にコードリポジトリ用 Dataset（`{username}/mlops-pipeline-src`）が作成済み
  - 中身: `src/` ディレクトリを zip してアップロード（初回のみ手動）

### 実行コマンド

```bash
# 依存インストール（初回のみ）
uv sync

# Notebook 生成 + Kaggle プッシュ
uv run python -m src usecase=push_notebook notebook.competition=titanic

# 生成されたファイルの確認
ls outputs/push_notebook/titanic/
# notebook.ipynb
# kernel-metadata.json
# README.md
# .gitignore
```

### `kernel-metadata.json` の主要フィールド

```json
{
  "id": "{username}/titanic-pipeline",
  "title": "titanic-pipeline",
  "code_file": "notebook.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": ["{username}/mlops-pipeline-src"],
  "competition_sources": ["titanic"]
}
```

---

## 実装アーキテクチャ

### 新規ファイル一覧

```
src/
  infrastructure/
    kaggle/
      environment.py        ← KaggleEnvironment（パス解決）
      __init__.py
  usecase/
    kaggle_notebook/
      push_notebook.py      ← PushNotebookUseCase
      notebook_renderer.py  ← Jinja2 テンプレートを ipynb に変換
      __init__.py

templates/
  notebook/
    pipeline.ipynb.j2       ← Jinja2 テンプレート（3セル構成）

conf/
  usecase/
    push_notebook.yaml      ← Hydra 設定

tests/
  infrastructure/
    kaggle/
      test_environment.py
      __init__.py
  usecase/
    kaggle_notebook/
      test_push_notebook.py
      __init__.py
```

### `KaggleEnvironment` 設計

```python
# src/infrastructure/kaggle/environment.py
class KaggleEnvironment:
    """Kaggle Notebook 環境の検出とパス解決を行うアダプター層。

    Hydra Config を直接書き換えず、実行時にパスを解決する薄いラッパー。
    is_kaggle_notebook() は KAGGLE_KERNEL_RUN_TYPE 環境変数で判定する。
    """

    @staticmethod
    def is_kaggle_notebook() -> bool:
        return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

    @staticmethod
    def resolve_input_root(dataset_slug: str) -> Path:
        """input データのルートを返す。
        Kaggle → /kaggle/input/{slug}
        ローカル → Path(dataset_slug)（呼び出し元がローカルパスを渡す）
        """
        if KaggleEnvironment.is_kaggle_notebook():
            return Path("/kaggle/input") / dataset_slug
        return Path(dataset_slug)

    @staticmethod
    def resolve_output_root() -> Path:
        """output データのルートを返す。
        Kaggle → /kaggle/working
        ローカル → Path(".")（呼び出し元が output_dir を絶対指定する）
        """
        if KaggleEnvironment.is_kaggle_notebook():
            return Path("/kaggle/working")
        return Path(".")
```

### `PushNotebookUseCase` 設計

```python
# src/usecase/kaggle_notebook/push_notebook.py
class PushNotebookUseCase:
    """Notebook 生成 → kernel-metadata.json 生成 → Kaggle プッシュ を担う。

    Kaggle API は Protocol で抽象化し、テスト時に Mock に差し替え可能。
    """

    def __init__(self, cfg: DictConfig, kaggle_api: KaggleApiPort) -> None: ...

    def execute(self) -> PushResult:
        output_dir = Path(self.cfg.output_dir) / self.cfg.notebook.competition
        output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_gitignore(output_dir)
        notebook_path = self._render_notebook(output_dir)
        metadata_path = self._generate_kernel_metadata(output_dir)
        self._push(output_dir)
        self._write_readme(output_dir, notebook_path, metadata_path)
        return PushResult(notebook_path=notebook_path, metadata_path=metadata_path)
```

### `conf/usecase/push_notebook.yaml`

```yaml
# @package _global_
usecase: push_notebook

notebook:
  competition: titanic            # Kaggle competition slug（input として追加）
  kernel_slug: titanic-pipeline   # Kaggle Kernel の URL slug
  src_dataset: mlops-pipeline-src # src/ を格納する Kaggle Dataset slug
  enable_gpu: false
  enable_internet: true

output_dir: outputs/push_notebook
```

### レイヤー構成（Clean Architecture 準拠）

```
main.py (presentation)
  └─ PushNotebookUseCase (usecase)
       ├─ KaggleApiPort (domain protocol)  ← テスト時は Mock
       │    └─ KaggleApiImpl (infrastructure)  ← 本番は Kaggle API
       └─ KaggleEnvironment (infrastructure adapter)
```

---

## コミット計画（RED → GREEN → REFACTOR）

| # | フラグ | コミットメッセージ |
|---|--------|-----------------|
| 1 | `--no-verify` | `[test] add KaggleEnvironment tests because of kaggle/local path resolution requirement` |
| 2 | 通常 | `[fix] implement KaggleEnvironment for input/output root resolution` |
| 3 | `--no-verify` | `[test] add PushNotebookUseCase tests because of notebook push workflow` |
| 4 | 通常 | `[fix] implement PushNotebookUseCase with Jinja2 template and kernel-metadata` |
| 5 | 通常 | `[fix] add usecase=push_notebook to main.py and conf` |
| 6 | 通常 | `[refactor] clean up types, docstrings, and imports` |
| 7 | 通常 | `[docs] add docs/manual/push-notebook.md` |

---

## テストケース一覧

### `tests/infrastructure/kaggle/test_environment.py`

fixture: `monkeypatch.setenv` / `monkeypatch.delenv` で環境変数を制御（副作用なし）

| テスト名 | 検証内容 | 前提 | 期待結果 |
|---------|---------|------|---------|
| `test_is_kaggle_notebook_false_when_env_absent` | 環境変数なし → ローカル判定 | `KAGGLE_KERNEL_RUN_TYPE` 未設定 | `False` |
| `test_is_kaggle_notebook_true_when_env_interactive` | Interactive → Kaggle判定 | `KAGGLE_KERNEL_RUN_TYPE=Interactive` | `True` |
| `test_is_kaggle_notebook_true_when_env_batch` | Batch → Kaggle判定 | `KAGGLE_KERNEL_RUN_TYPE=Batch` | `True` |
| `test_resolve_input_root_local` | ローカル → slug をそのまま Path に | 環境変数なし | `Path("titanic")` |
| `test_resolve_input_root_kaggle` | Kaggle → `/kaggle/input/titanic` | `KAGGLE_KERNEL_RUN_TYPE=Interactive` | `Path("/kaggle/input/titanic")` |
| `test_resolve_output_root_local` | ローカル → `Path(".")` | 環境変数なし | `Path(".")` |
| `test_resolve_output_root_kaggle` | Kaggle → `/kaggle/working` | `KAGGLE_KERNEL_RUN_TYPE=Interactive` | `Path("/kaggle/working")` |

### `tests/usecase/kaggle_notebook/test_push_notebook.py`

fixture: `tmp_path` で出力先分離・`MagicMock` で Kaggle API を差し替え（CI で通信なし）

| テスト名 | 検証内容 | 期待結果 |
|---------|---------|---------|
| `test_execute_creates_notebook_ipynb` | `notebook.ipynb` が生成される | ファイルが存在する |
| `test_generated_notebook_has_three_cells` | Notebook が3セルで構成される | `len(cells) == 3` |
| `test_generated_notebook_cell_types_are_code` | 全セルが `code` タイプ | `cell["cell_type"] == "code"` |
| `test_generated_notebook_cell1_contains_pip_install` | セル1に `pip install` が含まれる | `"pip install"` in cell1 source |
| `test_generated_notebook_cell2_contains_competition_slug` | セル2に competition slug が含まれる | `"titanic"` in cell2 source |
| `test_generated_notebook_cell3_contains_pipeline_run` | セル3に `PipelineUseCase` の呼び出しが含まれる | `"PipelineUseCase"` in cell3 source |
| `test_execute_creates_kernel_metadata_json` | `kernel-metadata.json` が生成される | ファイルが存在する |
| `test_kernel_metadata_has_required_fields` | JSON に必須フィールドが含まれる | `id`, `language`, `kernel_type` が存在する |
| `test_kernel_metadata_competition_sources` | competition が `competition_sources` に設定される | `["titanic"]` |
| `test_execute_calls_kernels_push` | Kaggle API `kernels_push()` が1回呼ばれる | `mock.assert_called_once()` |
| `test_execute_raises_runtime_error_on_auth_failure` | 認証失敗（SystemExit）→ RuntimeError | `RuntimeError` が上がる |
| `test_execute_creates_gitignore` | `.gitignore` が生成される | ファイルが存在する |
| `test_execute_creates_readme` | `README.md` が生成される | ファイルが存在する |

---

## DoD（完了条件）

- [ ] `uv run python -m src usecase=push_notebook notebook.competition=titanic` で Notebook が生成・プッシュされる
- [ ] Notebook セル3で `KaggleEnvironment` がパスを自動解決し、パイプライン全体が動く
- [ ] `kernel-metadata.json` に `competition_sources` と `dataset_sources` が設定される
- [ ] 全テストが CI（GitHub Actions + lefthook）で自動実行される
- [ ] TDD サイクル（RED → GREEN → REFACTOR）のコミット粒度が守られている
- [ ] `docs/manual/push-notebook.md` に手動実行手順が記載されている
- [ ] `outputs/push_notebook/{competition}/` に per-directory `.gitignore` が動的生成される
- [ ] `outputs/push_notebook/{competition}/README.md` にツリー構造が出力される

---

## 保留事項（今回スコープ外）

- Notebook 実行ステータスのポーリング（`kernels_status()`）— 初期実装はブラウザ確認
- `src/` を Kaggle Dataset に自動アップロードする `dataset push` ステップ — 初期実装は手動アップロード案内
- submission の自動提出（`competitions_submissions_upload()`）— 手動確認後に人が提出
