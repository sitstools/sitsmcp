# syntax=docker/dockerfile:1
FROM python:3.11-slim

# uv for dependency management (pinned binary from the official image)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Bake the embedding and reranker models into a stable cache path so workers start offline
ENV FASTEMBED_CACHE_PATH=/opt/fastembed_cache \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Install production dependencies first for better layer caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --extra prod --no-dev --frozen --no-install-project

# Pre-download the embedding model (~1.3 GB) at build time
RUN uv run python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-large-en-v1.5')"

# Pre-download the default reranker (~80 MB), the one open use gets
RUN uv run python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

# Application code and the pre-built database
COPY sitsmcp/ ./sitsmcp/
COPY gunicorn.conf.py ./
COPY sitsmcp.db ./sitsmcp.db

# Install the project itself into the environment
RUN uv sync --extra prod --no-dev --frozen

# Run as a non-root user with read access to the cache and database
RUN useradd --create-home --uid 1000 app \
    && chown -R app:app /app /opt/fastembed_cache
USER app

# Use the built virtualenv directly so startup does not re-sync via `uv run`
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["gunicorn", "-c", "gunicorn.conf.py", "sitsmcp.asgi:app"]
