# push_notebook / push_vertex_notebook の routing-only 化

## 背景

`conf/usecase/push_notebook.yaml` と `conf/usecase/push_vertex_notebook.yaml` に
`competition: titanic`, `kernel_slug: titanic-pipeline` 等の競技固有値がハードコードされており、
別コンペで `uv run python -m src usecase=push_notebook` を直接実行すると常に titanic 向けの値が使われてしまう
（caution.md の「usecase の config に インフラ固有の設定を混在させない」原則に違反）。

調査の結果、姉妹リポジトリ `../bird-clef-2026` で同種の修正（commit `924e863`）が行われていたが、
usecase yaml を routing-only にしただけで、`recipe=` から competition 固有 yaml を読み込むマージ処理が
追加されておらず、ドキュメントコメントに書かれた `notebook=all_after_download` という起動方法は実際には機能しない
（`run_push_notebook` に merge 呼び出しがなく、直接 CLI 実行時は `cfg.notebook` が `null` のまま）。
B 側では pipeline usecase 経由でのみ動作しており、単体実行は実質未対応のまま放置されていた。

また、調査中に `conf/competition/histopathologic/pipeline/full.yaml` の step 7
（`notebook: histopathologic_inference`）が文字列のまま渡されており、
`PushNotebookUseCase` が `cfg.notebook.competition` にアクセスする箇所で `AttributeError` になる
既存バグを発見した（テストなし・未実行のため検出されていなかった）。

## やること

1. `conf/usecase/push_notebook.yaml` / `push_vertex_notebook.yaml` を routing-only にする
   （`usecase: push_notebook` と `output_dir` のみ）。
2. `src/usecase/notebook/notebook_loader.py` を新設し、既存の
   `pipeline_loader.py` / `trainer_loader.py` と同じパターンで
   `load_notebook_recipe_cfg(cfg, conf_dir)` を実装する
   （`cfg.recipe` から `conf/competition/{competition.name}/notebook/{recipe}.yaml` をロードしてマージ）。
3. `src/presentation/runners.py` の `run_push_notebook` で `load_notebook_recipe_cfg` を呼び、
   単体実行（pipeline 経由でない直接 CLI 実行）でも competition 固有値が解決されるようにする。
   pipeline 経由（step で `notebook:` を直接 dict 指定）の場合は `cfg.notebook` が既に解決済みなので
   `recipe` が無ければ何もしない（既存の `load_pipeline_recipe_cfg` と同じ早期 return パターン）。
4. `conf/competition/titanic/notebook/all_after_download.yaml` /
   `conf/competition/titanic/notebook/inference_only.yaml` を新設し、
   旧 `push_notebook.yaml` / `push_vertex_notebook.yaml` にあった値を移植する。
5. `conf/competition/histopathologic/pipeline/full.yaml` の step 7 を
   文字列指定からインライン dict 指定に修正し、既存バグを解消する。
6. テスト:
   - `load_notebook_recipe_cfg` の単体テスト（recipe 指定あり/なし/存在しない場合）
   - `conf/competition/histopathologic/pipeline/full.yaml` を `build_step_configs` に通し、
     notebook step の cfg が dict 形状であることを検証する回帰テスト
7. `docs/manual/` に直接実行コマンド例を追記する。

## 対象外（スコープ外）

- `trainer_loader.py` / `inference_loader.py` / `pipeline_loader.py` の重複コードの統合
  （再利用性向上の余地はあるが、今回は push_notebook 単体の不具合修正にとどめる。別タスクで検討）
- BirdCLEF 固有の SeedFixer Protocol 化・input_dim ルールは別タスク（#2, #3）で対応する
