# 実行計画: データセットダウンロード実装（Kaggle対応）

## 目的

`data_from` 設定でインフラを切り替えられる汎用データダウンロード機能を実装する。
初回はKaggle対応のみだが、将来的にGCS・HuggingFace等への拡張を想定したClean Architectureで設計する。
コマンドはユースケースに対応させ、将来の preprocess / train / analyze 追加を容易にする。

## 背景

Kaggle APIの認証方式が新しくなっている（Legacy: kaggle.json → 新: KAGGLE_API_TOKEN 環境変数 or ~/.kaggle/access_token）ため、
新方式を優先した実装にする。

## 認証方式（新仕様）

| 優先度 | 方法 | 詳細 |
|--------|------|------|
| 1位（推奨） | 環境変数 | `export KAGGLE_API_TOKEN=xxxxx` |
| 2位 | トークンファイル | `~/.kaggle/access_token`（トークン文字列のみ） |
| 3位（Legacy） | JSONファイル | `~/.kaggle/kaggle.json`（旧方式） |

## アーキテクチャ

```
src/
├── __init__.py
├── main.py                              # CLI entry point: Hydra + ユースケース dispatch（python-dotenv で .env 自動ロード）
├── domain/
│   └── data/
│       ├── __init__.py
│       └── downloader.py               # Protocol: DataDownloader（抽象）
├── infrastructure/
│   └── downloader/
│       ├── __init__.py
│       ├── kaggle.py                   # KaggleDownloader(DataDownloader)
│       └── (将来: gcs.py, huggingface.py 等)
└── usecase/
    └── data_acquisition/
        ├── __init__.py
        └── download_dataset.py         # DownloadDatasetUseCase（抽象Downloaderに依存）

conf/
├── config.yaml                         # defaults: [usecase: download_dataset]
└── usecase/
    └── download_dataset.yaml           # data_from, output_dir, kaggle設定等

tests/
├── domain/
│   └── data/
│       └── test_downloader_protocol.py
├── infrastructure/
│   └── downloader/
│       ├── __init__.py
│       └── test_kaggle.py              # モックテスト（先に実装）
└── usecase/
    └── data_acquisition/
        ├── __init__.py
        └── test_download_dataset.py

docs/
├── manual/
│   └── kaggle-dataset-download.md      # ユーザー手動手順書
└── tasks/2026/Q1/0311_kaggle-download/
    ├── README.md（本ファイル）
    └── TEST_LOG_*.md
```

## 設計

### ドメイン: DataDownloader（抽象）

```python
# src/domain/data/downloader.py
from typing import Protocol
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DownloadResult:
    output_dir: Path
    files: list[Path]
    commit_hash: str   # DoD: CommitHash記録

class DataDownloader(Protocol):
    def download(self) -> DownloadResult: ...
```

### インフラ: KaggleDownloader

```python
# src/infrastructure/downloader/kaggle.py
class KaggleDownloader:
    def __init__(self, cfg: DictConfig) -> None: ...
    def download(self) -> DownloadResult: ...
```

### ユースケース: DownloadDatasetUseCase

```python
# src/usecase/data_acquisition/download_dataset.py
class DownloadDatasetUseCase:
    def __init__(self, downloader: DataDownloader) -> None: ...
    def execute(self) -> DownloadResult: ...
```

### エントリーポイント: main.py（ユースケース dispatch + DI）

```python
# src/main.py
@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # usecase= で切り替え可能
    # 将来: usecase=preprocess, usecase=train 等
    downloader = _resolve_downloader(cfg)
    DownloadDatasetUseCase(downloader).execute()
```

### Hydra Config

```yaml
# conf/config.yaml
defaults:
  - usecase: download_dataset
  - _self_

# conf/usecase/download_dataset.yaml
seed: 42
data_from: "kaggle"         # インフラ選択キー
output_dir: "data/2026/Q1/raw"
unzip: true
force: false

kaggle:
  mode: "dataset"           # "dataset" or "competition"
  dataset: "username/dataset-name"
  competition: null
```

## テスト方針

- `KaggleApiExtended` を `unittest.mock.MagicMock` でモック → CI上でも動作
- `pytest.mark.integration` で実認証テストをスキップ可能にする
- 各テストに **なぜ必要か** を docstring で明記

## 実行コマンド

```bash
# ユースケースを指定して実行（将来の拡張例）
uv run python -m src usecase=download_dataset
uv run python -m src usecase=preprocess   # 将来
uv run python -m src usecase=train        # 将来
```

## 作業ステップ（implementation.md準拠）

1. [x] ブランチ作成: `feature/kaggle-download`
2. [x] `docs/tasks/2026/Q1/0311_kaggle-download/README.md` 作成
3. [ ] CI自動化確認（type-check の src/ 対応）
4. [ ] テスト実装のみ（RED確認）→ TEST_LOG 保存
5. [ ] 実装（GREEN）
6. [ ] `docs/manual/kaggle-dataset-download.md` 作成
7. [ ] PR作成

## .env による認証管理

```bash
# .env.example をコピーしてトークンを設定（一度だけ）
cp .env.example .env
# .env を編集: KAGGLE_API_TOKEN=your-token-here
```

`main.py` で `python-dotenv` により `.env` を自動ロードするため、毎回 `export` 不要。
devenv.nix の `dotenv.enable = true` により devenv shell 入場時も自動ロード。

## 検証方法

```bash
# 初回セットアップ（一度だけ）
uv sync --extra kaggle --extra dev

# モックテスト（CI対象）
uv run pytest tests/ -v

# 手動実行（.env に KAGGLE_API_TOKEN 設定済みなら追加操作不要）
uv run python -m src usecase=download_dataset
```
