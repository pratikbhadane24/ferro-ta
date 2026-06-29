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
#
# CPU portability
# ---------------
# This image installs the PRE-BUILT ferro-ta wheel from PyPI — we do NOT
# recompile from sdist with `RUSTFLAGS=-C target-cpu=...`. The wheel is built
# at the manylinux baseline (x86-64-v1) and selects AVX2/AVX-512/NEON kernels
# at RUNTIME via CPU dispatch. One image therefore runs on any node — old or
# new CPU, x86_64 or arm64 — with no illegal-instruction (SIGILL) crashes.
# Pinning a target-cpu would be faster on a uniform fleet but would crash on
# any older/heterogeneous node, which is the opposite of broad coverage.
#
# Build this image for whichever arch your nodes use:
#   docker build --platform linux/amd64 -t ferro-ta-api .
#   docker build --platform linux/arm64 -t ferro-ta-api .   # Graviton/Ampere
# Both resolve a matching manylinux wheel — no Rust toolchain needed here.

FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies first (cache layer). No compiler is needed:
# ferro-ta, numpy, and pydantic-core all ship prebuilt wheels for linux
# x86_64 and aarch64.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Fail the build immediately if the wheel did not resolve for this arch
# (e.g. an exotic platform that fell back to an sdist build without Rust).
RUN python -c "import ferro_ta, numpy as np; ferro_ta.SMA(np.arange(10.0), 3); print('ferro_ta', ferro_ta.__version__, 'import OK')"

# Copy API source
COPY main.py ./

# Expose API port
EXPOSE 8000

ENV MAX_SERIES_LENGTH=100000

# Run with uvicorn (single worker; scale horizontally via Docker Compose / k8s)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
