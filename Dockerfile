# ────────────────────────────────────────────────────────────────────────────
# Build stage
# ────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ────────────────────────────────────────────────────────────────────────────
# Runtime stage
# ────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Install libGL required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY app/       ./app/
COPY configs/   ./configs/

# Default environment (override at runtime via -e or docker-compose env_file)
ENV CAMERA_RTSP_URL=rtsp://10.4.100.101:554/stream \
    MODEL_PATH=yolov8n.pt \
    LANE_CONFIG_PATH=configs/lanes_example.yaml \
    CONF_THRESHOLD=0.4 \
    IOU_THRESHOLD=0.5 \
    ALLOWED_CLASSES=car,truck,bus,motorcycle \
    DB_PATH=/data/counts.db

VOLUME ["/data"]

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
