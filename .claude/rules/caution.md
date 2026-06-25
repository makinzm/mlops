# 注意事項（過去の指摘事項）

このファイルはユーザーからの指摘をもとに、繰り返さないべき設計ミスや注意点をまとめたものです。

---

## ブランチ管理

### Squash Merge 済みブランチには継続作業しない

- **NG**: PR が Squash Merge されたブランチに追加コミットを積み続け、次の PR で大量のコンフリクトが発生する
- **OK**: PR が Squash Merge されたら、次の作業は必ず `origin/main` から新しいブランチを切る
  ```bash
  git fetch origin
  git checkout -b feat/next-task origin/main
  ```
- **Why**: Squash Merge は元ブランチのコミット履歴を持たない1コミットとして main に入る。元ブランチに追加コミットして PR を出すと、squash 前の全コミットが "新規変更" として扱われてコンフリクト地獄になる
- **How to apply**: セッション開始時に `git log --oneline origin/main..HEAD` で確認し、作業ブランチが既に main に squash 済みなら即座に新ブランチへ移行する

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

### `__main__.py` で CWD をプロジェクトルートに固定する

- **NG**: `src/__main__.py` に `os.chdir` なしのまま → mlops 外から実行すると `data/2026/Q1/raw/...` が CWD 基準で解決されてファイルが見つからない
- **OK**: `src/__main__.py` の冒頭で `os.chdir(Path(__file__).parent.parent)` を呼ぶ
- **Why**: Hydra の `config_path` は `Path(__file__)` 基準で絶対パス指定済みのため影響なし。data パスだけが CWD 依存になる
- **How to apply**: 新しいエントリーポイントを作るたびに必ず追加する

### データディレクトリは per-directory .gitignore で管理する（実行時生成）

- **NG**: ルート `.gitignore` に `data/**` パターンを書いて一括管理する（output_dir が変わったとき機能しない）
- **OK**: `output_dir` ごとに実行時 `.gitignore` を動的生成する。責務は以下に持たせる:
  - download: `GitRepositoryImpl.setup_data_dir(output_dir)`
  - EDA: `_setup_output_gitignore()` in analyzer
  - preprocess: `GitRepositoryImpl.setup_data_dir(cfg.output_dir)` in PreprocessUseCase
- **Why**: `output_dir` は conf で自由に変わるため、どこに出力しても自動的に機能する必要がある
- **How to apply**: 新しい usecase の output を追加するときは必ずこの一覧を確認し、.gitignore 生成を組み込む

### per-directory .gitignore を設計するときは保持対象を全て洗い出す

- **NG**: `!*.yaml` だけ追加して `*.html` / `*.md` を後から気づいて追加修正した
- **OK**: そのディレクトリで生成されうる全ファイル種別を確認してから書く
- 現プロジェクトの保持対象: `*.yaml`（metainfo, result）, `*.html`（pipeline_dag）, `*.md`（README）, `.gitkeep`, `.gitignore`, `*/`（サブディレクトリ）
- **How to apply**: `_DATA_DIR_GITIGNORE` / `_EDA_DIR_GITIGNORE` を変更するときは出力ファイル一覧と照合する

### .gitignore のクリーンアップ時は既知の除外対象を確認してから削除する

- **NG**: `.gitignore` の整理で `outputs/`（Hydra ログ）まで削除してしまい、Hydra の設定スナップショットが staging された
- **OK**: 各パターンを削除する前に「何のために書かれているか」を確認する。ツール固有の出力ディレクトリは絶対に消さない
- **Why**: `outputs/` は Hydra のデフォルト出力ディレクトリ。プロジェクト独自のパターンと区別できていなかった
- **How to apply**: `.gitignore` 変更 PR では削除したパターンの用途を必ずコメントに残す

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

### モデルの入力次元はハードコードせず関連設定から自動計算する

- **NG**: embedding 系モデルの `input_dim` を `1536` のように固定値で書く
  → embedding の次元（`embedding.dim`）や追加特徴量の有無（`save_logits` 等）が変わると
  学習時とは異なる次元で推論が実行され、Kaggle 提出が失敗する
- **OK**: `input_dim` は `embedding.dim + (追加特徴量の次元 if 有効化フラグ else 0)` のように、
  関連する設定値から学習・推論の両パスで動的に導出する
- **Why**: 姉妹プロジェクト（BirdCLEF 2026）で `save_logits=True` 時に `input_dim` を
  ハードコードしたまま放置し、次元不一致で提出が失敗した実績がある
- **How to apply**: モデルの入力/出力次元を設定値から計算する箇所を追加・変更するときは、
  学習側と推論側の両方で同じ計算式を使っているかをテストで確認する
  （`checkpoint['input_dim'] == 期待値` のようなアサーションを学習・推論両方のテストに置く）

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

## .gitignore パターン

### ~~`data/` ではなく `data/**` を使う~~（obsolete: per-directory 戦略に移行済み）

> この方針は廃止。ルート `.gitignore` に `data/**` を書くのではなく、
> 実行時に output_dir ごとへ per-directory `.gitignore` を動的生成する方針に移行した。
> 詳細は「データディレクトリは per-directory .gitignore で管理する」を参照。

### 既知のツール出力ディレクトリは必ず `.gitignore` に残す

| ディレクトリ | 生成元 | 理由 |
|------------|--------|------|
| `outputs/` | Hydra | `@hydra.main` が実行時に config スナップショットとログを保存 |
| `.pytest_cache/` | pytest | テストキャッシュ |
| `mlruns/` | MLflow | 実験ログ |
| `htmlcov/` | pytest-cov | カバレッジレポート |

- **NG**: `.gitignore` 整理で `outputs/` を消す → Hydra の設定スナップショット（`.hydra/config.yaml` 等）が staging される
- **OK**: ツール名とディレクトリの対応を把握したうえで削除判断する

---

## Hydra 設定

### conf/config.yaml に宣言していないキーは CLI から渡せない

- **NG**: `conf/config.yaml` に `recipe` キーを書かずに `uv run python -m src recipe=base` を実行する
  → `Key 'recipe' is not in struct` エラーになる
- **OK**: CLI で渡す予定のキーは `conf/config.yaml` に `recipe: null` 等として事前に宣言しておく
- **Why**: Hydra はルートの DictConfig を struct mode で管理するため、未定義キーを受け付けない
- **How to apply**: 新しい CLI パラメータ（`recipe=`, `trainer_name=` 等）を追加するときは
  `conf/config.yaml` への追記とセットで行う

### OutputResolver cv=False の出力形式はサブディレクトリ形式

- **NG**: `cv=False` 時に `{output_dir}/{node_id}.parquet`（フラットファイル）で出力する
  → `InferenceUseCase` が `{output_dir}/{node_id}/test.parquet` を期待するため FileNotFoundError
- **OK**: `cv=False` 時も `{output_dir}/{node_id}/test.parquet`（サブディレクトリ）で出力する
- **Why**: InferenceUseCase が `latest` 解決後に `test_out/test.parquet` を読むため、
  フォルダ構造が統一されている必要がある
- **How to apply**: OutputResolver の出力形式を変更するときは InferenceUseCase の読み込みパスも確認する

### config group のキーはグループ名以下に配置される

- **NG**: `conf/usecase/download_dataset.yaml` に `output_dir` を書いて `cfg.output_dir` でアクセスする
  （Hydra はデフォルトで `cfg.usecase.output_dir` に配置する）
- **OK**: ルートレベルで参照したいキーを持つ config ファイルには `# @package _global_` を先頭に追加する
- グループ名以下に置きたい設定（`cfg.downloader.*` など）は `@package _global_` 不要
- マージ結果は `OmegaConf.to_yaml(cfg)` で確認する

### 新しい usecase には必ず `conf/usecase/{name}.yaml` を作成する

- **NG**: `main.py` に `elif usecase_name == "preprocess":` を追加しただけで実行すると `Could not find 'usecase/preprocess'` エラーになる
- **OK**: usecase を追加するときは `conf/usecase/{name}.yaml`（最低でも `# @package _global_\nusecase: {name}` だけ）を必ず作成する
- **Why**: Hydra は `usecase=preprocess` を `conf/usecase/preprocess.yaml` として解決するため、ファイルがないとエラーになる
- **How to apply**: main.py に usecase 分岐を追加するときは conf とセットで追加する

### Hydra struct モードの DictConfig に新しいキーをマージできない

- **NG**: `OmegaConf.merge(hydra_cfg, external_yaml)` → `ConfigKeyError: Key 'job_id' is not in struct`（Hydra の DictConfig は struct mode で未定義キーを受け付けない）
- **OK**: `OmegaConf.to_container(cfg, resolve=True)` で plain dict に変換 → `OmegaConf.create()` で非 struct DictConfig を作ってからマージする:
  ```python
  base = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
  merged = DictConfig(OmegaConf.merge(base, OmegaConf.load(yaml_path)))
  ```
- **Why**: Hydra が生成した DictConfig は struct mode になっており、スキーマ外のキーを追加できない
- **How to apply**: 外部の yaml ファイル（competition 固有設定など）を Hydra cfg にマージするときは必ず to_container 経由にする

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

- **NG**: `git commit -m "test(RED): ..."` （pre-commit フックが ty / ruff で失敗する）
- **OK**: `git commit --no-verify -m "test(RED): ..."` （意図的に失敗している状態をコミットする）
- RED フェーズは `src/` が存在しないため import エラーが必然。フックをスキップして問題ない。
- GREEN フェーズ以降は通常通り `--no-verify` をつけない。

---

## コマンドの書き方

### extras を使わない。uv add --dev または uv add --group を使う

- **NG**: `uv run --extra dev pytest` / `uv run --extra kaggle python -m src.main`
- **OK**: `uv add pytest --dev` / `uv add kaggle --group kaggle` でインストール後、`uv run pytest` のみ
- `pyproject.toml` の `[project.optional-dependencies]` は使わず、`[dependency-groups]` で管理する
- CI でも `--extra` フラグなしで `uv run pytest` / `uv run ty check` を実行する

### python を直接実行しない。常に uv run python を使う

- **NG**: `python -c "..."` / `python script.py`
- **OK**: `uv run python -c "..."` / `uv run python script.py`
- プロジェクトの仮想環境を確実に使うために uv 経由で実行する。

### `uv run` に毎回 `--extra` をつけない

- **NG**: `uv run --extra kaggle --extra dev pytest tests/`
- **OK**: 初回に `uv sync --extra kaggle --extra dev` を一度だけ実行し、その後は `uv run pytest tests/` で十分
- ドキュメントでは「セットアップ」と「日常コマンド」を分けて記載する

---

## Kaggle Notebook

### Notebook push 前に必ず Dataset を更新する

- **NG**: 新しい config（例: `conf/competition/titanic/pipeline/inference_only.yaml`）を追加して Notebook push だけ行う → Kaggle 上で `FileNotFoundError: pipeline yaml が見つかりません`
- **OK**: Notebook push の前に `uv run python -m src usecase=update_source_dataset` で `mlops-pipeline-src` Dataset を更新する
- **Why**: Notebook は Kaggle 上の Dataset の中身を `/kaggle/input/` から読む。ローカルに conf を追加しても Dataset に push していなければ Kaggle 上には存在しない
- **How to apply**: パイプラインでは `update_source_dataset` ステップを `push_notebook` より前に配置する。手動実行時も同様の順序を守る

### Kaggle Dataset のパスは新旧2形式が存在する

- 旧形式: `/kaggle/input/{slug}/`
- 新形式: `/kaggle/input/datasets/{owner}/{slug}/`
- **NG**: テンプレートでパスをハードコードする
- **OK**: テンプレート（`templates/notebook/pipeline.ipynb.j2`）で両方を `os.path.exists` で検出し、存在する方を使う
- **How to apply**: テンプレートの `_DATASET_ROOT` 自動検出ロジックを変更しないこと

### シンボリックリンクのパスには `/src` サフィックスが必要

- **NG**: `_SRC_DATASET_PATH = '/kaggle/input/{slug}'` → `from src.xxx` が動かない（`No module named 'src'`）
- **OK**: `_SRC_DATASET_PATH = '/kaggle/input/{slug}/src'` → staging で `src/` がサブディレクトリとして配置されるため
- `os.path.lexists` を使う（`os.path.exists` だと壊れたシンボリックリンクを検出できない）
- 既存リンクを無条件削除してから再作成する

### ステージングに .gitignore をコピーしない

- **NG**: モデルディレクトリの `.gitignore`（`*` パターン）がステージングにコピーされる → Kaggle API がアップロード時に読み、全ファイルを除外 → 0 バイトの Dataset
- **OK**: `_copy_dir_to_staging` で `.gitignore` を除外する（`src/usecase/source_dataset/_staging.py`）
- **How to apply**: ステージングコピーのロジックを変更するときは `.gitignore` 除外を維持する

### 削除した Kaggle Dataset の slug は再利用できない

- **NG**: Kaggle 上で Dataset を削除して同じ slug で再作成 → `[Dataset no longer available]`
- **OK**: 新しい slug 名を使って作成する
- **How to apply**: Dataset slug を変更する場合は config ファイル（`conf/usecase/` 以下）も更新する

---

## 一時ファイル

### `/tmp` を使わない。カレントディレクトリに一時ファイルを置く

- **NG**: `/tmp/req_check.txt` のようにシステムの一時ディレクトリにファイルを書く
- **OK**: カレントディレクトリ（プロジェクトルート）に一時ファイルを置き、不要になったら削除する
- **Why**: `/tmp` はプロジェクトの外であり、管理が行き届かない。カレントディレクトリなら `.gitignore` や目視で管理できる
- **How to apply**: 一時ファイルが必要な場合はカレントディレクトリに作成し、使い終わったら削除する

---

## caution.md 運用

### 指摘を受けたルールは caution.md とプロジェクトルートの両方に追記する

- **NG**: `caution.md` だけ更新して、プロジェクトルート（`.claude/rules/caution.md`）への反映を忘れる。あるいはその逆。
- **OK**: 指摘を受けたら `.claude/rules/caution.md`（このファイル）に追記し、同時にプロジェクトの memory（`MEMORY.md` 等）にも反映する
- **Why**: caution.md はリポジトリにコミットされて共有される。memory は個人のセッション間で引き継がれる。両方に書くことで漏れを防ぐ
- **How to apply**: 指摘を受けたら必ず (1) このファイルに NG/OK 形式で追記 (2) MEMORY.md にサマリを追記、の2ステップを実行する
