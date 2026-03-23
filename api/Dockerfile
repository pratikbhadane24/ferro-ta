# ferro-ta API — Docker image
#
# Build:
#   docker build -t ferro-ta-api .
#
# Run:
#   docker run -p 8000:8000 ferro-ta-api
#
# Environment variables (override at runtime):
#   MAX_SERIES_LENGTH=100000   # maximum data-point count per request

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required to build ferro_ta (Rust is pre-compiled
# into the wheel, so only pip + wheel tooling is needed at runtime).
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first (cache layer)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy API source
COPY main.py ./

# Expose API port
EXPOSE 8000

ENV MAX_SERIES_LENGTH=100000

# Run with uvicorn (single worker; scale horizontally via Docker Compose / k8s)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
