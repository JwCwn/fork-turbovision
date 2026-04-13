FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python deps — timm==0.9.2 matches lRomul; scipy for peak detection.
# (E2E-Spot needs older timm but it's disabled by default so we prioritize lRomul.)
RUN pip install --no-cache-dir \
    fastapi uvicorn httpx pydantic opencv-python-headless cryptography \
    ultralytics supervision numpy "timm==0.9.2" scipy einops \
    "fiber @ git+https://github.com/rayonlabs/fiber.git@a41ab890708757140a3cf4aae8e5af57a8b03159"

# E2E-Spot repo (kept for legacy; disabled by default)
RUN git clone --depth 1 https://github.com/jhong93/spot.git /app/spot

# lRomul ball-action-spotting model source (MultiDimStacker)
RUN git clone --depth 1 https://github.com/lRomul/ball-action-spotting.git /tmp/lromul && \
    mkdir -p /app/lromul_src && \
    cp -r /tmp/lromul/src/models /app/lromul_src/ && \
    rm -rf /tmp/lromul

# Models are mounted at runtime from host:
#   /app/models/              (YOLO *.pt files)
#   /app/models/lromul/       (lRomul .pth checkpoint)
#   /app/models/e2e-spot/     (optional E2E-Spot checkpoint)
RUN mkdir -p /app/models /app/models/lromul /app/models/e2e-spot

COPY scorevision/miner/private_track/ ./scorevision/miner/private_track/
COPY scorevision/utils/logging.py scorevision/utils/schemas.py ./scorevision/utils/
RUN echo "" > scorevision/__init__.py && \
    echo "" > scorevision/miner/__init__.py && \
    echo "" > scorevision/utils/__init__.py

EXPOSE 8000

CMD ["uvicorn", "scorevision.miner.private_track.server:app", "--host", "0.0.0.0", "--port", "8000"]
