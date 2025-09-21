FROM python:3.12-slim

# Copier les binaires UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_TOOL_BIN_DIR=/usr/local/bin
ENV PATH="/app/.venv/bin:$PATH"

# Copier les fichiers de dépendances
COPY pyproject.toml uv.lock ./

# Installer les dépendances dans un virtualenv via UV
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Copier le code source après installation des dépendances
COPY src/ui /app/src/ui

# Exposer le port Streamlit
EXPOSE 8501

ENTRYPOINT []

CMD ["streamlit", "run", "src/ui/ui.py", "--server.port", "8501", "--server.address", "0.0.0.0"]