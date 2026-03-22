# src/ を Kaggle Dataset に自動アップロードする実装計画

- 作成日: 2026-03-16
- ブランチ: `feat/kaggle-notebook-push`
- 関連: PR #25

---

## 背景・目的

`docs/manual/push-notebook.md` では `src/` を Kaggle Dataset に登録する手順を「初回のみ手動」としていた。
しかし毎回のコード変更後にも更新が必要であり、手動手順はミスが起きやすい。

このタスクでは以下を自動化する:

1. **新規作成**: `src/` を Kaggle Dataset として初回登録する
2. **バージョン更新**: コード変更後に既存 Dataset の新バージョンをアップロードする

---

## 出来物のイメージ（ユーザー操作フロー）

### 初回: Dataset 新規作成

```bash
uv run python -m src usecase=create_source_dataset

# 出力:
# [INFO] Staging src/ to .staging/source_dataset_20260316_120000/
# [INFO] dataset-metadata.json written
# [INFO] Creating Kaggle Dataset: testuser/mlops-pipeline-src
# [INFO] Done. Dataset URL: https://www.kaggle.com/datasets/testuser/mlops-pipeline-src
# [INFO] Staging directory removed.
```

### コード変更後: バージョン更新

```bash
uv run python -m src usecase=update_source_dataset source_dataset.version_message="add target encoding"

# 出力:
# [INFO] Staging src/ to .staging/source_dataset_20260316_130000/
# [INFO] Updating Kaggle Dataset: testuser/mlops-pipeline-src (version: "add target encoding")
# [INFO] Done.
# [INFO] Staging directory removed.
```

### ステージングディレクトリの構成（成功時は自動削除）

```
.staging/
  source_dataset_20260316_120000/   ← タイムスタンプ付き
    src/                            ← プロジェクトの src/ をコピー
      __init__.py
      main.py
      domain/
      infrastructure/
      usecase/
    dataset-metadata.json           ← Kaggle Dataset のメタデータ
```

### .kaggleignore によるフィルタリング（プロジェクトルートに配置）

```gitignore
# .kaggleignore
__pycache__/
*.pyc
*.pyo
.venv/
*.egg-info/
.pytest_cache/
```

---

## アーキテクチャ設計

### 新規ファイル一覧

```
src/
  domain/
    repository/
      source_dataset.py         ← SourceDatasetRepository Protocol（新規）
  infrastructure/
    kaggle/
      source_dataset.py         ← KaggleSourceDatasetRepository（新規）
  usecase/
    source_dataset/
      create_source_dataset.py  ← CreateSourceDatasetUseCase（新規）
      update_source_dataset.py  ← UpdateSourceDatasetUseCase（新規）
      __init__.py

conf/
  usecase/
    create_source_dataset.yaml  ← Hydra 設定（新規）
    update_source_dataset.yaml  ← Hydra 設定（新規）

tests/
  domain/
    repository/
      test_source_dataset.py    ← Protocol の shape テスト（新規）
  infrastructure/
    kaggle/
      test_source_dataset.py    ← KaggleSourceDatasetRepository テスト（新規）
  usecase/
    source_dataset/
      test_create_source_dataset.py  ← UseCase テスト（新規）
      test_update_source_dataset.py  ← UseCase テスト（新規）
      __init__.py

.kaggleignore                   ← アップロード除外リスト（新規）
```

### `SourceDatasetRepository` Protocol（domain 層）

```python
# src/domain/repository/source_dataset.py
class SourceDatasetRepository(Protocol):
    """ソースコードの Dataset リポジトリ操作の抽象インターフェース。

    Kaggle 固有の実装を usecase 層から隠蔽する。
    将来 HuggingFace Hub / GCS 等への差し替えもこの Protocol を実装するだけで対応可能。
    """

    def create(self, staging_dir: Path, metadata: DatasetMetadata) -> None:
        """Dataset を新規作成する。"""
        ...

    def update_version(
        self, staging_dir: Path, metadata: DatasetMetadata, version_message: str
    ) -> None:
        """既存 Dataset の新バージョンを作成する。"""
        ...
```

### `DatasetMetadata` dataclass（domain 層）

```python
@dataclass(frozen=True)
class DatasetMetadata:
    title: str
    owner_slug: str   # Kaggle username
    dataset_slug: str # Dataset の slug 名（例: "mlops-pipeline-src"）
    license_name: str = "CC0-1.0"
```

### `KaggleSourceDatasetRepository` Infrastructure 実装

```python
# src/infrastructure/kaggle/source_dataset.py
class KaggleSourceDatasetRepository:
    """KaggleApi を使った SourceDatasetRepository の実装。

    create():
      1. staging_dir/ に dataset-metadata.json を書き出す
      2. KaggleApi.dataset_create_new(staging_dir) を呼ぶ

    update_version():
      1. staging_dir/ に dataset-metadata.json を書き出す
      2. KaggleApi.dataset_create_version(staging_dir, version_message) を呼ぶ
    """
```

### `CreateSourceDatasetUseCase` / `UpdateSourceDatasetUseCase`

```python
# src/usecase/source_dataset/create_source_dataset.py
class CreateSourceDatasetUseCase:
    """src/ を Kaggle Dataset として新規作成する。

    処理フロー:
    1. .staging/source_dataset_{timestamp}/ を作成する
    2. src/ を .kaggleignore でフィルタしながらステージングディレクトリにコピーする
    3. dataset-metadata.json を staging_dir/ に書き出す（Kaggle API が要求）
    4. SourceDatasetRepository.create() を呼ぶ
    5. 成功したらステージングディレクトリを削除する（失敗時は残す）
    """
```

### レイヤー構成（Clean Architecture 準拠）

```
main.py (presentation)
  ├─ CreateSourceDatasetUseCase (usecase)
  │    └─ SourceDatasetRepository (domain protocol)
  │         └─ KaggleSourceDatasetRepository (infrastructure)
  └─ UpdateSourceDatasetUseCase (usecase)
       └─ SourceDatasetRepository (domain protocol)
            └─ KaggleSourceDatasetRepository (infrastructure)
```

### `.staging/` ディレクトリ戦略

- プロジェクトルート直下の `.staging/` を使う（`/tmp` は使わない）
- ルート `.gitignore` に `.staging/` を追加する
- タイムスタンプ付きサブディレクトリ（`source_dataset_YYYYMMDD_HHMMSS/`）を使い並行実行の衝突を避ける
- **成功時**: staging dir を削除する
- **失敗時**: staging dir を残す（デバッグ用）

### Hydra 設定

```yaml
# conf/usecase/create_source_dataset.yaml
# @package _global_
usecase: create_source_dataset

source_dataset:
  src_dir: src                    # アップロード対象ディレクトリ
  dataset_slug: mlops-pipeline-src
  title: mlops-pipeline-src
  license_name: CC0-1.0
  kaggleignore: .kaggleignore     # フィルタリング設定ファイル（省略時はフィルタなし）

staging_dir: .staging
```

```yaml
# conf/usecase/update_source_dataset.yaml
# @package _global_
usecase: update_source_dataset

source_dataset:
  src_dir: src
  dataset_slug: mlops-pipeline-src
  title: mlops-pipeline-src
  license_name: CC0-1.0
  kaggleignore: .kaggleignore
  version_message: "update source code"  # CLI で上書き可

staging_dir: .staging
```

---

## コミット計画（RED → GREEN → REFACTOR）

| # | フラグ | コミットメッセージ |
|---|--------|-----------------|
| 1 | `--no-verify` | `[test] add SourceDatasetRepository protocol and UseCase tests because of source dataset upload automation` |
| 2 | 通常 | `[fix] implement SourceDatasetRepository protocol, KaggleSourceDatasetRepository, and UseCases` |
| 3 | 通常 | `[fix] add usecase=create/update_source_dataset to main.py and conf` |
| 4 | 通常 | `[refactor] clean up types, docstrings, and staging cleanup logic` |
| 5 | 通常 | `[docs] update docs/manual/push-notebook.md to reference new usecases` |

---

## テストケース一覧

### `tests/infrastructure/kaggle/test_source_dataset.py`

fixture: `tmp_path` で staging dir を代替・`MagicMock` で KaggleApi を差し替え

| テスト名 | 検証内容 | 期待結果 |
|---------|---------|---------|
| `test_create_writes_dataset_metadata_json` | `create()` が `dataset-metadata.json` を staging dir に書き出す | ファイルが存在し `id` フィールドが正しい |
| `test_create_calls_kaggle_api_dataset_create_new` | `create()` が `dataset_create_new()` を1回呼ぶ | `mock.assert_called_once()` |
| `test_create_raises_runtime_error_on_auth_failure` | 認証失敗（SystemExit）→ RuntimeError | `RuntimeError` が上がる |
| `test_update_version_writes_dataset_metadata_json` | `update_version()` が `dataset-metadata.json` を書き出す | ファイルが存在する |
| `test_update_version_calls_kaggle_api_create_version` | `update_version()` が `dataset_create_version()` を1回呼ぶ | `mock.assert_called_once()` |
| `test_update_version_passes_version_message` | `version_message` が API に渡される | `mock.call_args` に `version_message` が含まれる |

### `tests/usecase/source_dataset/test_create_source_dataset.py`

fixture: `tmp_path` で src_dir・staging_dir を用意・`MagicMock` で SourceDatasetRepository を差し替え

| テスト名 | 検証内容 | 期待結果 |
|---------|---------|---------|
| `test_execute_copies_src_to_staging` | `src/` の内容が staging dir にコピーされる | staging dir に `src/` 相当のファイルが存在する |
| `test_execute_calls_repository_create` | `SourceDatasetRepository.create()` が1回呼ばれる | `mock.create.assert_called_once()` |
| `test_execute_removes_staging_dir_on_success` | 成功時に staging dir が削除される | `staging_dir` が存在しない |
| `test_execute_keeps_staging_dir_on_failure` | 失敗時に staging dir が残る | `staging_dir` が存在する |
| `test_execute_respects_kaggleignore` | `.kaggleignore` でフィルタされたファイルがコピーされない | `__pycache__/` が staging に存在しない |
| `test_execute_uses_timestamp_subdir` | staging subdir がタイムスタンプ形式になる | `source_dataset_YYYYMMDD_HHMMSS` にマッチする |

### `tests/usecase/source_dataset/test_update_source_dataset.py`

| テスト名 | 検証内容 | 期待結果 |
|---------|---------|---------|
| `test_execute_copies_src_to_staging` | `src/` の内容が staging dir にコピーされる | ファイルが存在する |
| `test_execute_calls_repository_update_version` | `SourceDatasetRepository.update_version()` が1回呼ばれる | `mock.update_version.assert_called_once()` |
| `test_execute_passes_version_message` | `version_message` が repository に渡される | `mock.call_args` に `version_message` が含まれる |
| `test_execute_removes_staging_dir_on_success` | 成功時に staging dir が削除される | `staging_dir` が存在しない |
| `test_execute_keeps_staging_dir_on_failure` | 失敗時に staging dir が残る | `staging_dir` が存在する |

---

## DoD（完了条件）

- [ ] `uv run python -m src usecase=create_source_dataset` で `src/` が Kaggle Dataset に新規登録される
- [ ] `uv run python -m src usecase=update_source_dataset source_dataset.version_message="..."` でバージョン更新される
- [ ] `.staging/` がルート `.gitignore` に追加されている
- [ ] `.kaggleignore` がプロジェクトルートに存在する
- [ ] 成功時は `.staging/source_dataset_*/` が自動削除される
- [ ] 全テストが CI（GitHub Actions + lefthook）で自動実行される
- [ ] TDD サイクル（RED → GREEN → REFACTOR）のコミット粒度が守られている
- [ ] `docs/manual/push-notebook.md` の「初回のみ手動」の手順が新コマンドに置き換えられる
- [ ] `conf/config.yaml` に `source_dataset` / `staging_dir` キーが宣言されている（Hydra struct mode 対応）

---

## 保留事項（今回スコープ外）

- `src/` の差分のみを検出してアップロードする最適化
- アップロード済みバージョン一覧の確認コマンド（`kernels status` 相当）
