# Competition Config の管理とアーカイブ

## ディレクトリ構造

```
conf/
  competition/
    {name}/
      competition.yaml     # コンペのメタ情報（入力パス等）
      preprocess/
        base.yaml          # 共通前処理
        lgbm.yaml          # LightGBM 用（アンサンブル時）
        nn.yaml            # NN 用（アンサンブル時）

  archive/
    {YYYY}/Q{N}/
      {name}/              # 終了後に git mv で移動
        competition.yaml
        preprocess/
```

## 実行

```bash
# 前処理
uv run python -m src '+competition/titanic/preprocess=base'
uv run python -m src '+competition/titanic/preprocess=lgbm'

# ダウンロード
uv run python -m src usecase=download_dataset
```

## 新しいコンペを始めるとき

```bash
mkdir -p conf/competition/{name}/preprocess
cp conf/usecase/preprocess.yaml conf/competition/{name}/preprocess/base.yaml

cat > conf/competition/{name}/competition.yaml << 'EOF'
name: "{name}"
input_paths:
  - "data/YYYY/QN/raw"
EOF
```

`conf/config.yaml` の `competition:` もあわせて変更する。

```yaml
defaults:
  - competition: {name}/competition
```

## アーカイブ（コンペ終了後）

```bash
mkdir -p conf/archive/2026/Q1/
git mv conf/competition/titanic conf/archive/2026/Q1/titanic
git commit -m "archive: titanic"
```
