FROM python:3.11-slim

WORKDIR /app

# Install system deps + Poetry
RUN pip install --no-cache-dir poetry==2.4.1

# Copy dependency files first (layer caching)
COPY pyproject.toml poetry.lock ./

# Install dependencies (no virtualenv in container)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY experiments/ ./experiments/

EXPOSE 8501

CMD ["streamlit", "run", "src/visualization.py", "--server.address=0.0.0.0", "--server.port=8501"]
