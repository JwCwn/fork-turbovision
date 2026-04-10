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

RUN pip install --no-cache-dir \
    fastapi uvicorn httpx pydantic opencv-python-headless cryptography \
    ultralytics supervision numpy \
    "fiber @ git+https://github.com/rayonlabs/fiber.git@a41ab890708757140a3cf4aae8e5af57a8b03159"

# Models are mounted at runtime from host: /app/models -> see private_track_miner.py
RUN mkdir -p /app/models

COPY scorevision/miner/private_track/ ./scorevision/miner/private_track/
COPY scorevision/utils/logging.py scorevision/utils/schemas.py ./scorevision/utils/
RUN echo "" > scorevision/__init__.py && \
    echo "" > scorevision/miner/__init__.py && \
    echo "" > scorevision/utils/__init__.py

EXPOSE 8000

CMD ["uvicorn", "scorevision.miner.private_track.server:app", "--host", "0.0.0.0", "--port", "8000"]
