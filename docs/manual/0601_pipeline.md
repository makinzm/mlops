# 0601 — パイプライン（usecase=pipeline）

## 概要

前処理 → 学習 → 推論の一連のステップを `usecase=pipeline` で一括実行する。
各ステップは定義順に実行され、いずれかが失敗した場合は後続ステップは実行されない（fail-fast）。

---

## 前提条件

- `usecase=download`（または `usecase=download recipe=titanic`）が完了しており、
  `data/2026/Q1/raw/` 以下に生データが存在すること。

---

## 設定の確認

**`conf/competition/titanic/pipeline/all_after_download.yaml`**

```yaml
steps:
  - usecase: preprocess
    recipe: base
  - usecase: train
    recipe: lgbm
  - usecase: inference
    recipe: titanic_ensemble
```

各 step は `usecase` と `recipe` のペアで定義し、個別コマンドと同じ動作をする。

---

## 実行コマンド

```bash
# ダウンロード後の全ステップを一括実行
uv run python -m src usecase=pipeline recipe=all_after_download
```

---

## 出力先

各ステップの出力先は個別の usecase と同じ：

| ステップ | 出力先 |
|---------|--------|
| preprocess | `data/2026/Q1/processed/titanic_preprocess/<timestamp>/` |
| train | `models/titanic/titanic_lgbm/<timestamp>/` |
| inference | `data/2026/Q1/inference/titanic/titanic_inference/<timestamp>/` |

---

## 失敗時の動作

- いずれかのステップが失敗した場合、後続ステップは実行されない（fail-fast）
- エラーメッセージを確認し、各 usecase を個別に実行してデバッグする

個別実行コマンド：

```bash
# 前処理のみ
uv run python -m src usecase=preprocess recipe=base

# 学習のみ
uv run python -m src usecase=train recipe=lgbm

# 推論のみ
uv run python -m src usecase=inference recipe=titanic_ensemble
```

---

## カスタムパイプラインの作成

新しいパイプライン設定を `conf/competition/titanic/pipeline/` に追加することで、
任意のステップ組み合わせを定義できる。

例: Target Encoding を使ったパイプライン（`conf/competition/titanic/pipeline/target_encoding_pipeline.yaml`）:

```yaml
steps:
  - usecase: preprocess
    recipe: target_encoding
  - usecase: train
    recipe: lgbm_with_target_encoding
  - usecase: inference
    recipe: titanic_ensemble
```

実行:

```bash
uv run python -m src usecase=pipeline recipe=target_encoding_pipeline
```
