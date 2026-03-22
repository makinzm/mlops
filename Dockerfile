# Vertex AI カスタムトレーニングコンテナ
# ベースイメージ: Python 3.12 slim
FROM python:3.12-slim

# 必要なシステムパッケージ（LightGBM のビルド依存）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv のインストール
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 依存関係ファイルを先にコピー（レイヤーキャッシュ活用）
COPY pyproject.toml uv.lock ./

# 本番依存のみインストール
RUN uv sync --no-dev

# アプリケーションコードをコピー
COPY src/ ./src/
COPY conf/ ./conf/
COPY scripts/ ./scripts/

# エントリーポイントを実行可能にする
RUN chmod +x /app/scripts/vertex_entrypoint.py

# CWD をプロジェクトルートに固定（Hydra の相対パス解決のため）
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["uv", "run", "python", "/app/scripts/vertex_entrypoint.py"]
