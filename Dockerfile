# Image for the heart disease prediction API.
# Build:  docker build -t heart-api .
# Run:    docker run -p 8000:8000 heart-api
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install only what the API needs (a smaller, faster image than the full
# training requirements). Copied first so Docker can cache this layer.
COPY requirements-api.txt .
RUN pip install --upgrade pip && pip install -r requirements-api.txt

COPY src/ ./src/
COPY models/ ./models/

# Run as a non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
