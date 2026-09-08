FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY contracts ./contracts

RUN pip install --no-cache-dir .

RUN useradd -m appuser && mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD curl -f http://127.0.0.1:8000/health/live || exit 1

CMD ["uvicorn", "garment_grader.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
