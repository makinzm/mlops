# 注意事項（過去の指摘事項）

このファイルはユーザーからの指摘をもとに、繰り返さないべき設計ミスや注意点をまとめたものです。

---

## ファイル管理

### ドキュメントのコマンドは全箇所を実際に実行して確認する

- **NG**: 一箇所だけ直して「他も直っている」と判断する
- **OK**: `grep` 等で同種のコマンドを全ファイル・全箇所洗い出し、すべて実行確認してからコミットする
- 関連コマンドの一部だけ修正して残りに漏れがある状態でコミットしない

### `-m package` で実行できるよう __main__.py を作る

- **NG**: `src/main.py` だけ作って `uv run python -m src` が動かない状態にする
- **OK**: `src/__main__.py` を作り `main()` を呼ぶ。README に書いたコマンドは必ず動作確認する
- task README に書いたコマンド例は必ず手元で実行して確認する

### データディレクトリは .gitignore + .gitkeep で管理する

- **NG**: `data/` を `.gitignore` するだけでディレクトリ自体が git に入らない。また `data/2026/Q1/raw/` のように出力先をハードコードした `.gitkeep` を repo に入れる
- **OK**: `output_dir` は conf で自由に変わるため、**ダウンロード実行時に動的に** `.gitkeep` と `.gitignore` を生成するロジックを実装に組み込む
- その責務は `GitRepository.setup_data_dir()` に持たせ、downloader から DI で使う

### git 操作は GitRepository Protocol に集約する

- **NG**: `KaggleDownloader` が `subprocess.run(["git", ...])` を直接呼ぶ
- **OK**: `src/domain/repository/git.py` に `GitRepository` Protocol を定義し、`GitRepositoryImpl` に実装を閉じる
- commit hash 取得・データディレクトリ初期化など git に関わる操作はすべて `GitRepository` 経由にする

---

## メタルール

### ファイルを修正する前に必ず公式ドキュメントを確認する

- **NG**: Webサーチ結果を読む前にファイルを書き換える。間違った情報を元に実装する。
- **OK**: `gh` コマンド（例: `gh repo view Kaggle/kaggle-api --json ...`）や WebFetch/WebSearch で公式ドキュメントを確認してから実装する。
- ライブラリの認証・APIの仕様変更は特に確認が必要。思い込みで実装しない。
- URL付きのインラインコメントを残す場合は、必ず実在するURLを確認してから書く。嘘のURLは厳禁。

### 詰まったら2回で報告し、必ずログに残す

- 同じ問題に2回対処してもブロックされたら、すぐにユーザーへ状況を報告する
- 報告の前に `docs/tasks/YYYY/Q/MMDD_<title>/STUCK_LOG_YYYYMMDD_HHMMSS.md` を作成する
- ログには以下を含める:
  - 何をしようとしていたか
  - 試みた内容（1回目・2回目）と得られたエラー
  - 自分の仮説と次に必要なこと
- ログを作らずに口頭だけで報告するのはNG。記録が残らない。

### 指摘を受けたら必ず caution.md を更新する

- ユーザーから設計・実装・ドキュメント・コマンド等に関する指摘を受けた場合、**言われなくても** このファイルに追記する
- 指摘内容を「NG → OK」形式で記録し、次回以降同じミスを繰り返さない
- `caution.md` の更新は実装コミットとは別に、指摘を受けた直後に行う

---

## ドキュメント

### マニュアルには「識別子の調べ方」など操作に必要な情報を書く

- **NG**: `dataset: "owner/dataset-name"` とだけ書いて識別子の取得方法を省略する
- **OK**: URL から識別子を取得する具体的な手順をマニュアルに記載する
- ユーザーが迷わないよう、URL 例と識別子の対応を示す

### マニュアルには「出力先」を明記する

- **NG**: "ダウンロードされます" と書いてどこに保存されるか省略する
- **OK**: `output_dir`（デフォルト: `data/2026/Q1/raw/`）に保存される、と具体的に記載する
- `.gitignore` 対象であることも合わせて明記する

---

## 依存管理

### kaggle 等の中核ライブラリは main deps に入れる

- **NG**: `[dependency-groups]` の `kaggle` グループに入れて `uv sync --group kaggle` が必要にする
- **OK**: `[project.dependencies]` に入れて `uv sync` だけで動くようにする
- `uv sync` 単体で動くことが理想。覚えることを最小にする。

---

## アーキテクチャ

### インフラ層を直接呼ばない

- **NG**: CLI → `KaggleDownloader` を直接呼ぶ
- **OK**: CLI（presentation） → UseCase → Infrastructure という Clean Architecture のレイヤーを守る
- DI は CLI 層で行い、UseCase は抽象（Protocol）に依存させる

### usecase の config に インフラ固有の設定を混在させない

- **NG**: `conf/usecase/download_dataset.yaml` に `kaggle: {mode: dataset, ...}` を書く
- **OK**: `conf/downloader/kaggle.yaml` に kaggle 固有設定を分離し、usecase config は `output_dir / unzip / force` のみ持つ
- Clean Architecture: UseCase 層は Infrastructure の存在を知らない。設定も同様。

### インフラ選択は設定で抽象化する

- **NG**: `KaggleDownloader` を直接インスタンス化してハードコード
- **OK**: `data_from: "kaggle"` のような設定キーで実装を選択し、将来 GCS / HuggingFace 等への差し替えを容易にする
- 拡張性を意識し、特定ベンダーに密結合したクラス名をエントリーポイントに露出させない

### ユースケースはコマンドに対応させる

- **NG**: `main.py` がすべての処理を持ち、コマンドが増えるにつれて肥大化する
- **OK**: Hydra の config group（`usecase=download_dataset`）でユースケースを切り替え
- 将来 `usecase=preprocess`, `usecase=train`, `usecase=analyze` を追加しやすい構造にする

---

## 設定・環境変数

### `.env` で認証情報を管理する

- **NG**: ドキュメントに `export KAGGLE_API_TOKEN=xxxxx` と毎回手動実行を案内する
- **OK**: `.env.example` を用意し、`python-dotenv` で自動ロードする
- `devenv.nix` に `dotenv.enable = true` を追加して devenv shell 入場時も自動ロードする
- `.env` は `.gitignore` 済みであることを前提とする

### ライブラリの認証 env var は思い込みで書かない

- **NG**: `kaggle` ライブラリ用に `KAGGLE_API_TOKEN` を使うよう案内する（これは `kagglehub` 用）。あるいは `KAGGLE_USERNAME`+`KAGGLE_KEY` を案内する（これは Legacy 方式）
- **OK**: `gh repo view Kaggle/kaggle-api` 等で公式 README を確認し、正確な方法をドキュメント化する
- 公式ドキュメント（https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md）では以下の優先順:
  1. `export KAGGLE_API_TOKEN=xxx`（新方式・`~/.kaggle/settings.yaml` の `username:` と `key:` から生成）
  2. `~/.kaggle/access_token`（ファイル保存方式）
  3. `~/.kaggle/kaggle.json`（Legacy）
- ユーザーに一番シンプルな方法（`~/.kaggle/access_token`）をドキュメント化する

---

## ライブラリの import 時副作用

### try/except は副作用が発生する場所を正確に包む

- **NG**: `self.api.authenticate()` を `try/except SystemExit` で包む
  （`kaggle/__init__.py` は import 時に `api.authenticate()` を実行するため、
  `self.api.authenticate()` に到達する前に sys.exit() が呼ばれる）
- **OK**: `from kaggle.api.kaggle_api_extended import KaggleApi` の import 文自体を
  `try/except SystemExit` で包む
- ライブラリの副作用（import 時初期化・認証）は **import 文の行** で発生する。
  `__init__.py` 等を確認して副作用の発生場所を特定してから try/except を置く。

---

## TDD サイクル

### RED フェーズのコミットは --no-verify をつける

- **NG**: `git commit -m "test(RED): ..."` （pre-commit フックが mypy / ruff で失敗する）
- **OK**: `git commit --no-verify -m "test(RED): ..."` （意図的に失敗している状態をコミットする）
- RED フェーズは `src/` が存在しないため import エラーが必然。フックをスキップして問題ない。
- GREEN フェーズ以降は通常通り `--no-verify` をつけない。

---

## コマンドの書き方

### extras を使わない。uv add --dev または uv add --group を使う

- **NG**: `uv run --extra dev pytest` / `uv run --extra kaggle python -m src.main`
- **OK**: `uv add pytest --dev` / `uv add kaggle --group kaggle` でインストール後、`uv run pytest` のみ
- `pyproject.toml` の `[project.optional-dependencies]` は使わず、`[dependency-groups]` で管理する
- CI でも `--extra` フラグなしで `uv run pytest` / `uv run mypy` を実行する

### python を直接実行しない。常に uv run python を使う

- **NG**: `python -c "..."` / `python script.py`
- **OK**: `uv run python -c "..."` / `uv run python script.py`
- プロジェクトの仮想環境を確実に使うために uv 経由で実行する。

### `uv run` に毎回 `--extra` をつけない

- **NG**: `uv run --extra kaggle --extra dev pytest tests/`
- **OK**: 初回に `uv sync --extra kaggle --extra dev` を一度だけ実行し、その後は `uv run pytest tests/` で十分
- ドキュメントでは「セットアップ」と「日常コマンド」を分けて記載する
