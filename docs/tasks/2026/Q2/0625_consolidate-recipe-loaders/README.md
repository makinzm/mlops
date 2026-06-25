# trainer_loader / inference_loader / pipeline_loader / notebook_loader の共通化

## 背景

`conf/competition/{name}/{subdir}/{recipe}.yaml` を Hydra cfg にマージするロジックが
以下 4 ファイルにほぼ同一実装でコピペされている:

- `src/usecase/training/trainer_loader.py`（recipe 省略時は training/ 配下を全件ロード）
- `src/usecase/inference/inference_loader.py`（recipe 省略時は inference/ 配下を全件ロード）
- `src/usecase/preprocessing/pipeline_loader.py`（recipe 省略時は preprocess/ 配下を全件ロード）
- `src/usecase/pipeline/pipeline_loader.py`（recipe 必須・単一ファイルのみ）
- `src/usecase/notebook/notebook_loader.py`（recipe 省略時は cfg をそのまま返す・単一ファイル）

3 種類の振る舞い（複数ロード+フォールバック / 単一必須 / 単一省略可）に共通する
「to_container で struct 回避 → OmegaConf.merge → 見つからなければ利用可能一覧つきエラー」
を `src/usecase/_recipe.py` に集約し、既存 4 ファイルを薄いラッパーにする。

## やること

1. `src/usecase/_recipe.py` に以下の2関数を実装する:
   - `load_recipe_cfgs(cfg, subdir, conf_dir, *, fallback_key=None, empty_dir_message=None, label="recipe") -> list[DictConfig]`
     （trainer_loader / inference_loader / preprocessing.pipeline_loader 用）
   - `load_single_recipe_cfg(cfg, subdir, conf_dir, *, required=False, required_message=None, label="recipe") -> DictConfig`
     （usecase/pipeline/pipeline_loader / notebook_loader 用）
2. 既存 4 ファイルを上記関数を呼ぶだけの薄いラッパーに書き換える。
   関数名・シグネチャ・公開 API は変更しない（呼び出し側 `src/presentation/runners.py` は無修正）。
3. エラーメッセージの文言は既存テストが `match=` で検証している部分文字列
   （"xgboost", "training", "前処理設定が見つかりません" 等）を変えない。
4. 既存テスト（trainer_loader / preprocessing.pipeline_loader / notebook_loader）は無修正で全て green を維持する。
5. `inference_loader.py` にはテストが存在しなかったため、このタイミングで追加する
   （`_recipe.py` の振る舞いを inference 側からも検証する）。

## 対象外（スコープ外）

- 各 usecase の呼び出し側（runners.py）のロジック変更
- pipeline_loader 以外の usecase への新規 recipe マージ機能の追加
