# Cricket private-track miner (subnet 44) — GPU / CUDA image.
# Runs the TrackNet ball + keypoint models and the physics bundle-adjustment
# pipeline on the GPU. Requires nvidia-container-toolkit on the host and
# `docker run --gpus all ...`.
#
# CPU-only fallback (slower; OCR + TrackNet ~1-2 min/clip): swap the base image
# for `python:3.11-slim`, drop the python3.10 apt block below, and replace the
# torch line with:
#     pip install --no-cache-dir torch torchvision
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    libgl1 \
    libglib2.0-0 \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

RUN python -m pip install --upgrade pip

# PyTorch (CUDA 12.4 build) — not on the default PyPI index.
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Inference + serving deps.
RUN pip install --no-cache-dir \
    fastapi uvicorn httpx pydantic cryptography \
    "numpy>=1.26,<2.3" "opencv-python-headless>=4.10" scipy \
    "rapidocr-onnxruntime>=1.3" "onnxruntime>=1.17" \
    "fiber @ git+https://github.com/rayonlabs/fiber.git@a41ab890708757140a3cf4aae8e5af57a8b03159"

# Miner code + vendored cricket pipeline (incl. model weights under cricket/).
COPY scorevision/miner/private_track/ ./scorevision/miner/private_track/
COPY scorevision/utils/logging.py scorevision/utils/schemas.py ./scorevision/utils/
RUN echo "" > scorevision/__init__.py && \
    echo "" > scorevision/miner/__init__.py && \
    echo "" > scorevision/utils/__init__.py

EXPOSE 8000

CMD ["uvicorn", "scorevision.miner.private_track.server:app", "--host", "0.0.0.0", "--port", "8000"]
