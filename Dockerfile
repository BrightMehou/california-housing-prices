FROM python:3.12-slim


# Installer Poetry via pip
RUN pip install poetry

WORKDIR /app

COPY pyproject.toml poetry.lock /app/
RUN poetry install --no-root --without dev

# Copy the source code into the container.
COPY src/ml /app/src/ml
COPY src/api /app/src/api


# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
CMD poetry run python src/ml/train.py && poetry run uvicorn 'src.api.app:app' --host=0.0.0.0 --port=8000
