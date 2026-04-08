FROM python:3.11-slim

# Metadata
LABEL maintainer="LogAnalysisEnv"
LABEL description="OpenEnv: Autonomous Log Analysis & Incident Response"
LABEL version="1.0.0"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required for HF Spaces)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY environment/ ./environment/
COPY app.py .
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# Ownership
RUN chown -R appuser:appuser /app
USER appuser

# Expose port for HF Spaces
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]