ARG PYTHON_VERSION=3.14.2
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

FROM base AS build

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_PYTHON_PREFERENCE=only-system

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-install-project

COPY . ./
RUN uv sync --frozen

FROM base AS runtime

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

EXPOSE 80

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=build /app ./
