このプロジェクトは、MLOpsの実装を目的としています。

# AIへの必須指示

**作業開始前に `.claude/rules/caution.md` を必ず一読すること。**
このファイルにはユーザーから過去に指摘された設計ミス・コマンドの誤り・注意事項がまとめられている。
読まずに作業を始めると同じ指摘が繰り返される。また、作業中も忘れないよう定期的に確認すること。

まずはじめに、ローカル環境から簡単にKaggleのデータセットをダウンロードするためのコードを提供し、最終的にモデルの監視やデプロイメントまでをカバーする予定です。

# DoD

すべてのPRは次を満たす必要があります。

- [ ] Red -> Green -> RefactorのサイクルがCommitの粒度で行われていること。
- [ ] すべてのコードを書く前にテストが書かれていて、なぜそのテストが必要かあとから読んでもわかるように説明されていること。
- [ ] 自動化される前に、手動での動作確認方法が docs/manual/<task_name>.md に記載されていること。
- [ ] すべてのテストは、lefthookもしくは github actionsで自動化されていること。
- [ ] 全てのコードは、Seedで再現性が保たれていること。
- [ ] 全てのコード特に、リモートへ反映されるコードは当該GitのCommitHashが記録されていること。
- [ ] 時系列やIDをベースとする前処理や学習を行う場合は、データリークが起きないように、適切な方法でデータを分割していること。
- [ ] 全ての変数はHydraのConfigで管理されていること。

# TODO

毎回の作業ごとにTODOリストを更新していきます。

## 1st Phase: Kaggle

- [ ] devenv.nix, lefthook を利用して 1st phaseの開発環境を構築するためのドキュメントの作成。
- [ ] Kaggle APIを使用して、ローカル環境からKaggleのデータセットをダウンロードするコードの実装。
- [ ] ダウンロードしたデータセットをローカル環境で前処理するためのコードの実装。
- [ ] ローカル環境で開発したPythonコードをKaggle Notebookに移行するためのガイドラインの作成。
- [ ] Kaggle Notebookの内容をローカル環境でスクリプトで実行できるようにするためのコードの実装。
- [ ] ローカル環境で学習を行い、Kaggle Notebookで同じコードを実行して結果を比較するためのテストコードの実装。
- [ ] モデルごとに外れているデータポイントに注目できるようなコードの実装。
- [ ] モデルの予測結果を可視化するためのコードの実装。
- [ ] 学習済みのモデルで予測を行い、Kaggleで提出するためのコードの実装。
- [ ] モデルの学習をGCPのVertex AIで行うためのコードの実装。
- [ ] 上記の内容を自動化するための設計を作成し、Github Actionsで実装する。

## TBD

# Directory Structure

ディレクトリ構造は次のような内容を想定しています。

```
.
├── README.md
├── .gitignore
├── devenv.nix
├── .github
│   └── workflows
│       └── ci.yml
├── data
│   └── YYYY/Q/<dataset_name>/
|                  ├── raw/
|                  ├── processed/
|                  └── external/
├── models/
|      ├── general/
|      └── original/
|             ├── <competition_name or task_name>/
|                           ├── YYYYMMDD_HHMMSS/
|                                   ├── model.pkl
|                                   ├── config.yaml
|                                   ├── training.log
|                                   ├── model_card.md
|
├── docs/
│   ├── manual/
│   │   ├── how-to-start-kaggle-competition.md
|
├── notes/
│   ├── YYYY-MM-DD.md
|
├── src/
|   ├── main.py
│   ├── domain/
│   │   ├── data/
│   │        ├── images/
│   │        ├── tabular/
│   │        └── text/
│   │   ├── model/
│   │   ├── preprocessing/
│   │   └── postprocessing/
│   ├── usecase/
|          ├── preprocessing/
|          ├── training/
|          ├── inference/
|          └── evaluation/
│   └── infrastructure/
│       ├── kaggle/
│       ├── gcp/
│       └── aws/
└── tests/  # same structure as src/
    ├── domain/
    ├── usecase/
    └── infrastructure/
```
