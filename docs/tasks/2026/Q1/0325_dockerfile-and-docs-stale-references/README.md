# Dockerfile 移動 + ドキュメント陳腐化修正

## 背景

PR #34 (Clean Architecture Naming 準拠) で `vertex_entrypoint.py` → `remote_entrypoint.py` にリネームされたが、Dockerfile とマニュアルが未更新。Dockerfile のビルドが壊れている。

## 修正内容

| ファイル | 変更内容 |
|---------|---------|
| `Dockerfile` → `docker/Dockerfile` | 移動 + `vertex_entrypoint.py` → `remote_entrypoint.py` に修正 |
| `scripts/docker_push.sh` | `-f docker/Dockerfile` 追加 |
| `docs/manual/1001_vertex-ai-training.md` | usecase 名修正 (`vertex_train` → `remote_train`) + Dockerfile パス修正 |

## 検証方法

1. `grep -r "vertex_entrypoint" .` で残存参照がないこと
2. マニュアル内のコマンドが整合すること
3. lefthook（ruff, ty, pytest）が通ること
