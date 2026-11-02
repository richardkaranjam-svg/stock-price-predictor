# Antigravity Quant V3+ Production Container
FROM python:3.11-slim

# Install system dependencies for LightGBM & curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pinned requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose web dashboard & webhook API port
EXPOSE 5000

# Health check against Flask /health route
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command: launch web server & API
CMD ["python", "app.py"]
