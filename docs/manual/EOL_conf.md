# Competition Config の管理とアーカイブ手順

## ディレクトリ構造

```
conf/
  competition/               # アクティブなコンペ
    {competition_name}/
      competition.yaml       # コンペのメタ情報（名前・入力パス等）
      preprocess/            # 前処理設定（アンサンブルのモデルごとに複数可）
        base.yaml            # 共通ベース前処理
        lgbm.yaml            # LightGBM 用（スケーリング不要、カテゴリはそのまま等）
        nn.yaml              # NN 用（スケーリングあり、Embedding等）
        stacking.yaml        # スタッキング用の OOF 特徴量前処理
      model/                 # 将来: モデル設定
      ensemble/              # 将来: アンサンブル設定

  archive/                   # 終了したコンペ（構造は competition/ と同じ）
    {YYYY}/Q{N}/
      {competition_name}/
        competition.yaml
        preprocess/
          ...
```

## アンサンブルと前処理の対応

アンサンブルの各モデルは前処理が異なる場合があるため、`preprocess/` 以下に**モデルごとのファイル**を用意する。

| ファイル | 用途 |
|---------|------|
| `base.yaml` | 全モデル共通の前処理（欠損補完、カラム選択等）|
| `lgbm.yaml` | GBDTモデル用（スケーリング不要、カテゴリはLabel Encoding等）|
| `nn.yaml` | ニューラルネット用（StandardScaler、Embedding等）|
| `stacking.yaml` | スタッキング用（Out-of-Fold 特徴量を生成する前処理）|

各ファイルは `from_job` で `base.yaml` の出力を引き継ぐことができる。

```yaml
# preprocess/lgbm.yaml の例
inputs:
  - id: base_features
    from_job: titanic_preprocess_base   # base.yaml の出力を引き継ぐ
    output_id: train_out
steps:
  - id: lgbm_encoded
    polars:
      method: label_encode             # LightGBM 用カテゴリ変換
      columns: [Sex, Embarked]
  - id: lgbm_out
    output:
      columns: [PassengerId, Survived, Pclass, Sex, Age, FamilySize, Fare, Embarked, label]
      format: parquet
      cv: true
targets: [lgbm_out]
```

## 実行方法

```bash
# ベース前処理
uv run python -m src 'usecase=competition/titanic/preprocess/base'

# LightGBM 用前処理
uv run python -m src 'usecase=competition/titanic/preprocess/lgbm'

# NN 用前処理
uv run python -m src 'usecase=competition/titanic/preprocess/nn'
```

## アーカイブ手順

コンペ終了後（提出完了・結果確定後）に以下の手順でアーカイブする。

```bash
# 1. アーカイブ先ディレクトリを作成
mkdir -p conf/archive/2026/Q1/

# 2. git mv で移動（履歴を保持）
git mv conf/competition/titanic conf/archive/2026/Q1/titanic

# 3. コミット
git commit -m "archive: titanic コンペの設定をアーカイブ"
```

アーカイブ後もファイルは git 履歴で参照できるため、削除はしない。

## 新しいコンペを始めるとき

```bash
# ディレクトリ作成
mkdir -p conf/competition/{new_competition}/preprocess

# base.yaml をテンプレートからコピー
cp conf/usecase/preprocess.yaml conf/competition/{new_competition}/preprocess/base.yaml

# competition.yaml を作成
cat > conf/competition/{new_competition}/competition.yaml << EOF
name: "{new_competition}"
input_paths:
  - "data/YYYY/QN/raw"
EOF
```
