# Vertex AI カスタムトレーニングコンテナ
#
# このイメージは依存パッケージのみを含む。
# src/ や conf/ はベイクせず、GCS 経由で渡す。
# → deps が変わらない限り再ビルド不要。
#
# ビルド: ./scripts/docker_push.sh
FROM python:3.12-slim

# 必要なシステムパッケージ（LightGBM のビルド依存）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv のインストール
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 依存関係ファイルのみコピー（レイヤーキャッシュ活用）
COPY pyproject.toml uv.lock README.md ./

# 本番依存のみインストール（自パッケージはビルドしない）
RUN uv sync --no-dev --no-install-project

# エントリーポイントだけコピー（これは固定なのでイメージに含める）
COPY scripts/vertex_entrypoint.py ./scripts/vertex_entrypoint.py

# src/ は GCS から取得するため PYTHONPATH で解決する
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

ENTRYPOINT ["uv", "run", "python", "/app/scripts/vertex_entrypoint.py"]
