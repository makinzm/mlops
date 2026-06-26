#!/usr/bin/env bash
# PreToolUse hook: Write/Edit 時にディレクトリ構造が MLOps 規約に従っているかチェックする。
#
# 違反ケース:
#   1. src/ 直下に domain/ usecase/ infrastructure/ presentation/ 以外のディレクトリを作成
#   2. プロジェクトルート直下に規約外の新規ディレクトリを作成
#
# 違反時は exit 2 でブロックし、正しい配置先を提示する。

set -euo pipefail

# ツール入力を stdin から受け取る
INPUT=$(cat)

# file_path を抽出（Write は file_path、Edit は file_path）
FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('file_path', d.get('path', '')))
except Exception:
    print('')
" 2>/dev/null)

[ -z "$FILE_PATH" ] && exit 0

# リポジトリルートを取得（対象ディレクトリが未存在の場合は親を遡る）
_find_repo_root() {
    local dir="$1"
    while [ "$dir" != "/" ]; do
        if git -C "$dir" rev-parse --show-toplevel &>/dev/null; then
            git -C "$dir" rev-parse --show-toplevel
            return 0
        fi
        dir=$(dirname "$dir")
    done
    return 1
}

# ファイルの親ディレクトリから存在する最近傍を探す
_TARGET_DIR=$(dirname "$FILE_PATH")
while [ "$_TARGET_DIR" != "/" ] && [ ! -d "$_TARGET_DIR" ]; do
    _TARGET_DIR=$(dirname "$_TARGET_DIR")
done

REPO_ROOT=$(_find_repo_root "$_TARGET_DIR") || exit 0

# 相対パスへ変換
REL_PATH="${FILE_PATH#$REPO_ROOT/}"
[ "$REL_PATH" = "$FILE_PATH" ] && exit 0  # リポジトリ外のファイル

TOP_DIR=$(echo "$REL_PATH" | cut -d'/' -f1)
SECOND_DIR=$(echo "$REL_PATH" | cut -d'/' -f2)

# ── チェック 1: src/ 直下のディレクトリ ──────────────────────────────────────
if [ "$TOP_DIR" = "src" ] && [ -n "$SECOND_DIR" ]; then
    case "$SECOND_DIR" in
        domain|usecase|infrastructure|presentation|__init__.py|__main__.py|main.py|_*|*.py)
            : # 許可
            ;;
        *)
            echo "❌ [Directory Guard] src/${SECOND_DIR}/ は Clean Architecture 規約外です。" >&2
            echo "" >&2
            echo "   src/ 直下に作成できるのは以下のみです:" >&2
            echo "     domain/          — エンティティ・値オブジェクト・ビジネスロジック" >&2
            echo "     usecase/         — ユースケース（アプリケーションサービス）" >&2
            echo "     infrastructure/  — DB・外部API・ファイルシステムの実装" >&2
            echo "     presentation/    — CLI・API エンドポイント・ランナー" >&2
            echo "" >&2
            echo "   作成しようとしたファイル: ${REL_PATH}" >&2
            echo "   → 適切な層のサブディレクトリに配置し直してください。" >&2
            exit 2
            ;;
    esac
fi

# ── チェック 2: プロジェクトルート直下の新規ディレクトリ ──────────────────────
ALLOWED_ROOT_DIRS="src tests docs conf data models scripts templates terraform docker notebooks .github .claude devenv.nix devenv.yaml devenv.lock lefthook.yml mille.toml pyproject.toml uv.lock requirements.txt README.md CLAUDE.md .gitignore outputs reports remote_jobs_history"

# ドットファイル・既存ディレクトリ・拡張子付きファイルは除外
if [[ "$TOP_DIR" != .* ]] && [[ "$TOP_DIR" != *.* ]] && [ "$TOP_DIR" = "$REL_PATH" -o -n "$SECOND_DIR" ]; then
    if [ ! -d "$REPO_ROOT/$TOP_DIR" ] && [ ! -f "$REPO_ROOT/$TOP_DIR" ]; then
        KNOWN=false
        for d in $ALLOWED_ROOT_DIRS; do
            [ "$TOP_DIR" = "$d" ] && KNOWN=true && break
        done
        if [ "$KNOWN" = "false" ]; then
            echo "⚠️  [Directory Guard] ルート直下への新規ディレクトリ '${TOP_DIR}/' は規約外の可能性があります。" >&2
            echo "" >&2
            echo "   許可されているルートディレクトリ:" >&2
            echo "     src/  tests/  docs/  conf/  data/  models/  scripts/  templates/" >&2
            echo "     terraform/  docker/  notebooks/  .github/" >&2
            echo "" >&2
            echo "   作成しようとしたファイル: ${REL_PATH}" >&2
            echo "   → 既存のディレクトリ配下に配置してください。" >&2
            exit 2
        fi
    fi
fi

exit 0
