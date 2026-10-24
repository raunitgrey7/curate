# syntax=docker/dockerfile:1.7
# Stage 1: resolve the locked environment with uv (CPU-only torch wheels via pyproject index).
FROM ghcr.io/astral-sh/uv:0.13.5-python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-editable

# Stage 2: train the serving bundle so the runtime image starts instantly and needs no network.
FROM builder AS trainer
ARG CURATE_DATASET=movielens_100k
ENV PATH="/app/.venv/bin:$PATH" CURATE_DATA_DIR=/app/data CURATE_ARTIFACT_DIR=/app/artifacts
COPY data/CHECKSUMS.sha256 ./data/
RUN python -m curate.cli data && python -m curate.cli train --dataset ${CURATE_DATASET}

# Stage 3: minimal runtime, non-root user, only the virtualenv and the trained artifacts.
FROM python:3.13-slim-bookworm AS runtime
RUN groupadd --system curate && useradd --system --gid curate --home /app --shell /usr/sbin/nologin curate \
    && apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder --chown=curate:curate /app/.venv /app/.venv
COPY --from=trainer --chown=curate:curate /app/artifacts /app/artifacts
ENV PATH="/app/.venv/bin:$PATH" \
    CURATE_ARTIFACT_DIR=/app/artifacts \
    CURATE_DATA_DIR=/app/data \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2
USER curate
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD curl -fsS http://127.0.0.1:8000/health || exit 1
CMD ["uvicorn", "curate.serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--log-config", "/dev/null"]
