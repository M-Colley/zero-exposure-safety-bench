FROM python:3.12-slim-bookworm

ENV ZESB_OFFLINE_ONLY=1 \
    HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml LICENSE README.md ./
COPY configs ./configs
COPY src ./src

RUN pip install --no-cache-dir . \
    && mkdir -p /input /models /output \
    && chown -R 65532:65532 /app /input /models /output

USER 65532:65532

VOLUME ["/input", "/models", "/output"]

ENTRYPOINT ["zesb", "evaluate"]
