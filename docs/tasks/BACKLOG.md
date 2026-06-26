# Backlog

ルール:
- ID は `TASK-NNN`（3桁ゼロ埋め）で発行し、削除・再利用しない
- ステータス: `[ ]` 未着手 / `[~]` 進行中 / `[x]` 完了
- 完了したタスクは削除せず末尾に残す（履歴として機能させる）

---

## 未着手


  - **前提**: PR #53 マージ後に新ブランチで対応
  - **mille**: `src_usecase.name_deny` に `"remote"` を追加（`"vertex"` はすでにある）
  - **リネーム対象**:
    - `src/usecase/training/remote_train.py` → `cloud_train.py`
    - `src/usecase/training/remote_submit.py` → `cloud_submit.py`
    - `src/usecase/training/remote_download.py` → `cloud_download.py`
    - `src/presentation/runners.py`: `run_remote_train` → `run_cloud_train`、`run_vertex_submit` → `run_cloud_submit`、`run_vertex_download` → `run_cloud_download`
    - `src/presentation/registry.py`: キー `"remote_train"` / `"vertex_submit"` / `"vertex_download"` → `"cloud_train"` / `"cloud_submit"` / `"cloud_download"`
    - `conf/usecase/`: `vertex_submit.yaml`、`vertex_download.yaml`、`vertex_train.yaml`、`create_vertex_models.yaml`、`upload_vertex_models.yaml`、`push_vertex_notebook.yaml` → マージ後に実物確認してリネーム案確定
  - **参照箇所も含めて全置換**: pipeline.py 内の文字列参照、コメント、テストも対象

---

## 完了

- [x] **TASK-001** `AudioTrainer` を `runners.py` に接続する。commit `cbb742a`
- [x] **TASK-002** chezmoi: `pre-task-estimate.sh` にセッション初回 BACKLOG.md 自動注入を追加。グローバル CLAUDE.md にルール #7 追記。dot_claude commit `411a669` / chezmoi commit `7db8a3f`、両方 push 済。
- [x] **TASK-003** 実行環境の切り替え設計を整理・ドキュメント化する。`docs/manual/execution-env.md` 作成。commit `7d25f9d`
- [x] **TASK-004** `conf/executor/gcp_vertex.yaml` / `ray_local.yaml` の「フォールバックのみ」状態を解消する。`ExecutorFactory` を NotImplementedError / ValueError に変更。commit `7d25f9d`
- [x] **TASK-005** chezmoi: ループ検出しきい値 40→150、Stop フック後継続ルール追加。dot_claude commit `5c16848` / `c2ab994`
- [x] **TASK-006** chezmoi: dot_claude 変更は事前確認不要のルール追加。dot_claude commit `8f9d564`
- [x] **TASK-007** usecase 層からインフラ名（remote / vertex）を除去し mille ルールを追加。PR #54。
