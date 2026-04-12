FROM python:3.11-slim

LABEL maintainer="LogAnalysisEnv"
LABEL description="OpenEnv: Autonomous Log Analysis & Incident Response"
LABEL version="3.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL files needed by openenv validate
COPY pyproject.toml .
COPY uv.lock .
COPY openenv.yaml .
COPY README.md .

# Copy application code
COPY environment/ ./environment/
COPY server/ ./server/
COPY tests/ ./tests/
COPY app.py .
COPY inference.py .
COPY benchmark.py .
COPY ui.html .

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]