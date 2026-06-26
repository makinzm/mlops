# Backlog

ルール:
- ID は `TASK-NNN`（3桁ゼロ埋め）で発行し、削除・再利用しない
- ステータス: `[ ]` 未着手 / `[~]` 進行中 / `[x]` 完了
- 完了したタスクは削除せず末尾に残す（履歴として機能させる）

---

## 未着手

- [x] **TASK-001** `AudioTrainer` を `runners.py` に接続する
  - `conf/competition/audio_example/training/efficientnet_b0.yaml` の `trainer_type: audio` を `trainer.type: audio` に修正
  - `src/presentation/runners.py` の `run_train` に `elif trainer_type == "audio": trainer = AudioTrainer()` を追加
  - 回帰テスト追加

- [x] **TASK-002** chezmoi: BACKLOG.md をセッション開始時に自動注入する Hook を追加
- [x] **TASK-005** chezmoi: ループ検出しきい値を 40→150 に引き上げ（Stop フック対応後も継続ルール追加）
- [x] **TASK-006** chezmoi: dot_claude 変更は事前確認不要のルールを追加
  - 副作用: リモートへの push が承認なしで実行される
  - 追加理由: feature ブランチ push のたびに loop 検出が入る
  - 拒否時の代替: 現状通り都度承認
  - 変更先: `~/.local/share/chezmoi/dot_claude/settings.json` → `chezmoi apply`

- [ ] **TASK-003** 実行環境の切り替え設計を整理・ドキュメント化する
  - local / GCP Vertex（CPU・GPU） / AWS（未実装）の切り替え方を `docs/manual/execution-env.md` にまとめる
  - `conf/executor/gcp_vertex.yaml` と `conf/cloud/vertex.yaml` の役割の違いを明記
  - AWS 対応の設計判断（やる/やらない）を ADR に残す

- [ ] **TASK-004** `conf/executor/gcp_vertex.yaml` と `conf/executor/ray_local.yaml` の「フォールバックのみ」状態を解消する
  - 現状: `ExecutorFactory` がこれらを受け取っても `LocalExecutor` に落ちる
  - GCP Vertex executor を本実装するか、明示的に「未実装」エラーにするか判断する

---

## 完了

- [x] **TASK-001** `AudioTrainer` を `runners.py` に接続する。commit `cbb742a`
- [x] **TASK-002** chezmoi: `pre-task-estimate.sh` にセッション初回 BACKLOG.md 自動注入を追加。グローバル CLAUDE.md にルール #7 追記。dot_claude commit `411a669` / chezmoi commit `7db8a3f`、両方 push 済。
- [x] **TASK-005** chezmoi: ループ検出しきい値 40→150、Stop フック後継続ルール追加。dot_claude commit `5c16848` / `c2ab994`
- [x] **TASK-006** chezmoi: dot_claude 変更は事前確認不要のルール追加。dot_claude commit `8f9d564`
