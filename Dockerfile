FROM python:3.13-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock LICENSE README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev --no-editable

FROM python:3.13-slim-bookworm AS runtime

ENV ZESB_OFFLINE_ONLY=1 \
    HOME=/tmp \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app /input /models /output \
    && chown -R 65532:65532 /app /input /models /output

WORKDIR /app

COPY --from=builder --chown=65532:65532 /app/.venv /app/.venv
COPY --chown=65532:65532 configs /app/configs

USER 65532:65532

VOLUME ["/input", "/models", "/output"]

ENTRYPOINT ["zesb", "evaluate"]
