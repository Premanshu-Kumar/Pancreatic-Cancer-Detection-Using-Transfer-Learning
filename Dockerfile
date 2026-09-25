# =============================================================================
# Multi-Stage Dockerfile for Pancreatic Cancer Detection Application
# Compatible with Docker, Docker Compose, and Hugging Face Spaces (Port 7860/8000)
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build Dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install virtualenv and build dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production Runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    APP_HOST=0.0.0.0

# Install runtime dependencies required by OpenCV and ReportLab
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python dependencies from builder
COPY --from=builder /opt/venv /opt/venv

# Create unprivileged application user
RUN groupadd -g 10001 appuser && \
    useradd -u 10000 -g appuser -m -s /bin/bash appuser

# Copy application source code
COPY . /app

# Ensure proper write permissions for runtime directories
RUN mkdir -p /app/app/uploads /app/results /app/models /app/data && \
    chown -R appuser:appuser /app

USER appuser

# Expose FastAPI service port (8000: local, 7860: HF Spaces, 10000: Render)
EXPOSE 8000 7860 10000

# Health check to ensure API is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

# Launch uvicorn web server
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
