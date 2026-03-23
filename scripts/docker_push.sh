#!/bin/bash
# Docker イメージをビルドして Artifact Registry に push するスクリプト。
#
# 使い方:
#   ./scripts/docker_push.sh
#
# .env の GCP_PROJECT / GCP_REGION から push 先を自動決定する。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE が見つかりません。" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [ -z "${GCP_PROJECT:-}" ]; then
  echo "ERROR: .env に GCP_PROJECT が未設定です。" >&2
  exit 1
fi

REGION="${GCP_REGION:-asia-northeast1}"
IMAGE_URI="${REGION}-docker.pkg.dev/${GCP_PROJECT}/mlops/training:latest"

echo "=== Docker build ==="
echo "Image: ${IMAGE_URI}"
docker build -t "$IMAGE_URI" "$PROJECT_ROOT"

echo ""
echo "=== Docker push ==="
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
docker push "$IMAGE_URI"

echo ""
echo "=== 完了 ==="
echo "Image URI: ${IMAGE_URI}"
