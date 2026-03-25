# Timeline: Dockerfile 移動 + ドキュメント陳腐化修正

## 2026-03-25

### 作業開始

- ブランチ `fix/dockerfile-and-docs-stale-references` を `origin/main` から作成
- 対象ファイルの確認完了

### 実装

- テスト不要（Dockerfile 移動 + ドキュメント修正のみ、Python コード変更なし）
- Dockerfile を `docker/Dockerfile` に移動し、entrypoint パスを修正
- `scripts/docker_push.sh` のビルドコマンドを更新
- `docs/manual/1001_vertex-ai-training.md` の陳腐化した参照を修正
