FROM dustynv/pytorch:2.7-r36.4.0-cu128-24.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    python3-venv \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_PYTHON_PREFERENCE=only-system \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --inexact --no-install-project --no-install-package torch --no-install-package numpy

COPY . ./
RUN uv sync --frozen --inexact --no-install-package torch --no-install-package numpy

EXPOSE 80
