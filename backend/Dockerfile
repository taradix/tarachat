FROM dustynv/pytorch:2.7-r36.4.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    python3-venv \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_PYTHON_PREFERENCE=only-system

WORKDIR /app

# Venv inherits system torch and numpy (torch compiled against numpy 1.x)
RUN uv venv --system-site-packages

COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-install-project --no-install-package torch --no-install-package numpy

COPY . ./
RUN uv sync --frozen --no-install-package torch --no-install-package numpy

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 80
