

FROM python:3.11-slim

# Prevent Python from writing pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Install build/runtime dependencies required for native packages (gmpy2, cryptography, azure libs, etc.)
# Keep as few packages as needed, remove build deps after pip install to reduce image size.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        pkg-config \
        libgmp-dev \
        libmpfr-dev \
        libmpc-dev \
        libssl-dev \
        libffi-dev \
        ca-certificates \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a requirements file with the pinned/required dependencies (ensures reproducible build even if host doesn't provide requirements.txt)
RUN cat > /tmp/requirements.txt <<'REQ'
fastapi>=0.95.0
uvicorn>=0.18.0
pydantic>=2.0
openai>=2.6.0
anthropic>=0.1.0
python-dotenv>=1.0.0
aiohttp>=3.11.0
rich>=13.0.0
redis>=4.6.0
gmpy2>=2.1.0
azure-ai-contentsafety>=1.0.0
azure-core>=1.30.0
REQ

# Install Python dependencies
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-deps --requirement /tmp/requirements.txt && \
    rm -f /tmp/requirements.txt

# Remove build dependencies to reduce image size but keep runtime libs (e.g., libgmp)
RUN apt-get purge -y --auto-remove \
        build-essential \
        gcc \
        g++ \
        pkg-config \
        libgmp-dev \
        libmpfr-dev \
        libmpc-dev \
        libssl-dev \
        libffi-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Copy application source
# It is recommended that your project root contains the app files (e.g., agent.py, modules/, requirements.txt if used)
COPY . /app

# Create non-root user and ensure ownership
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose the application port
EXPOSE 8000

# Default command: run uvicorn with multiple workers suitable for production.
# Override number of workers and port with env vars when running the container.
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

