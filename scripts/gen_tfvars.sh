#!/bin/bash
# .env から terraform/terraform.tfvars を自動生成するスクリプト。
#
# 使い方:
#   ./scripts/gen_tfvars.sh
#
# .env が single source of truth。terraform.tfvars を手で編集しないこと。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"
TFVARS_FILE="$PROJECT_ROOT/terraform/terraform.tfvars"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE が見つかりません。" >&2
  echo "1000_gcp-initial-setup.md のセクション 10 を参照して .env を作成してください。" >&2
  exit 1
fi

# .env を読み込む（コメント行と空行を無視）
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# 必須変数のチェック
missing=()
[ -z "${GCP_PROJECT:-}" ] && missing+=("GCP_PROJECT")

if [ ${#missing[@]} -gt 0 ]; then
  echo "ERROR: .env に以下の変数が未設定です:" >&2
  printf "  - %s\n" "${missing[@]}" >&2
  exit 1
fi

# bucket_name のデフォルト: {project_id}-mlops-staging
BUCKET_NAME="${GCP_BUCKET_NAME:-${GCP_PROJECT}-mlops-staging}"

cat > "$TFVARS_FILE" << EOF
# このファイルは scripts/gen_tfvars.sh で自動生成されています。
# 手動で編集しないでください。変更は .env に対して行ってください。
#
# 再生成: ./scripts/gen_tfvars.sh

project_id  = "${GCP_PROJECT}"
region      = "${GCP_REGION:-asia-northeast1}"
bucket_name = "${BUCKET_NAME}"
EOF

echo "生成完了: $TFVARS_FILE"
