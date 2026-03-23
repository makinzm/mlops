# mypy → ty 移行

## 概要

型チェッカーを mypy から ty（Astral 製）に完全移行する。

## 変更対象

1. **pyproject.toml**: `[tool.mypy]` セクション → `[tool.ty]` セクションへ置換。`mypy` / `types-*` スタブを dev deps から削除し、`ty` を追加。
2. **lefthook.yml**: `mypy` コマンド → `ty check` コマンドへ置換。
3. **.github/workflows/ci.yml**: `uv run mypy` → `uv run ty check` へ置換。
4. **ソースコード**: `# type: ignore[...]` コメント → `# ty: ignore[...]` へ変換（エラーコードもマッピング）。
5. **.dockerignore / .kaggleignore**: `.mypy_cache` → `.ty_cache` 参照があれば更新。

## エラーコードマッピング（主要）

| mypy | ty |
|------|-----|
| import-untyped | unresolved-import |
| arg-type | invalid-argument-type |
| assignment | invalid-assignment |
| union-attr | possibly-missing-attribute |
| misc | （個別対応） |
| no-untyped-def | （Ruff ANN で対応） |
| call-overload | invalid-argument-type |
| unused-ignore | unused-ignore-comment |

## 方針

- `ty check` を実行して現状のエラーを確認し、必要に応じて `--add-ignore` で既存エラーを抑制。
- stubs パッケージ（`types-PyYAML`, `pandas-stubs` 等）は ty がネイティブで解決できるか確認し、不要なら削除。
- pydantic plugin 依存はないため移行ブロッカーなし。
