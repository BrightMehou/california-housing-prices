FROM python:3.13-slim

# Copie des binaires UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_TOOL_BIN_DIR=/usr/local/bin
ENV PATH="/app/.venv/bin:$PATH"

# Copier le pyproject.toml et uv.lock avant d'installer les dépendances
COPY pyproject.toml uv.lock ./

# Installer les dépendances via UV
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Copier le code source après installation des dépendances
COPY src/ /app/src/
COPY data/ /app/data/

# Exposer le port
EXPOSE 8000

# Entrypoint
ENTRYPOINT []
