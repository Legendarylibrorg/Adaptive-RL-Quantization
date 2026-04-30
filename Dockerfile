# syntax=docker/dockerfile:1

FROM python:3.12.10-slim-bookworm@sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db

ARG INSTALL_EXTRAS=""
ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface/transformers \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN set -eux; \
    groupadd --system --gid "${APP_GID}" app; \
    useradd --system --uid "${APP_UID}" --gid app --home-dir /tmp --shell /usr/sbin/nologin app; \
    python -m venv "${VIRTUAL_ENV}"

COPY . /app

RUN set -eux; \
    python scripts/verify_hashes.py --output /tmp/requirements-ci.txt; \
    python -m pip install --require-hashes -r /tmp/requirements-ci.txt; \
    if [ -n "${INSTALL_EXTRAS}" ]; then \
        python -m pip install --no-build-isolation ".[${INSTALL_EXTRAS}]"; \
    else \
        python -m pip install --no-build-isolation .; \
    fi; \
    mkdir -p /app/outputs /tmp/.cache /tmp/huggingface /tmp/matplotlib; \
    chown -R app:app /app/outputs /tmp/.cache /tmp/huggingface /tmp/matplotlib

USER app:app

CMD ["adaptive-rl-quant", "--config", "config.e2e_smoke.json"]
