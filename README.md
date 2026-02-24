# carCount – Two-Lane Vehicle Counter

A **FastAPI** web application that reads an RTSP surveillance-camera stream,
detects and tracks vehicles with **YOLOv8 + ByteTrack**, and counts them
individually for **two lanes** travelling in the same direction.

---

## Table of Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Local Setup](#local-setup)
4. [Configuration](#configuration)
5. [Lane Calibration](#lane-calibration)
6. [Network Access to the Camera](#network-access-to-the-camera)
7. [Docker](#docker)
8. [API Reference](#api-reference)
9. [Running Tests](#running-tests)
10. [GPU Acceleration](#gpu-acceleration)
11. [Project Structure](#project-structure)

---

## Features

| Feature | Details |
|---|---|
| Live RTSP ingest | OpenCV VideoCapture; auto-reconnects on failure |
| Vehicle detection | YOLOv8 (default `yolov8n.pt`, configurable) |
| Multi-object tracking | Ultralytics built-in ByteTrack (stable IDs across frames) |
| Two-lane counting | Configurable polygons + counting line; direction-aware |
| Deduplication | Each track ID counted only once per crossing |
| Persistence | Count events stored in SQLite via aiosqlite |
| Web dashboard | HTML page with live counts, auto-refresh every 3 s |
| REST API | `/api/stats` (JSON) and `/api/frame` (annotated JPEG) |
| Docker ready | Multi-stage Dockerfile + docker-compose.yml |

---

## Requirements

- Python **3.10+**
- (Optional) NVIDIA GPU with CUDA for faster inference

---

## Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/rcret/carCount.git
cd carCount

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and set CAMERA_RTSP_URL, MODEL_PATH, etc.

# 5. Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open your browser at **http://localhost:8000** to see the dashboard.

> **Tip:** The app starts successfully even when the camera is unreachable.
> The background worker will keep retrying the connection.  
> `/api/stats` always returns a valid JSON response.

---

## Configuration

Copy `.env.example` to `.env` and adjust the values:

| Variable | Default | Description |
|---|---|---|
| `CAMERA_RTSP_URL` | `rtsp://10.4.100.101:554/stream` | RTSP URL of the surveillance camera |
| `MODEL_PATH` | `yolov8n.pt` | Path to YOLO weights (auto-downloaded if missing) |
| `LANE_CONFIG_PATH` | *(empty)* | Path to lane geometry YAML/JSON (see `configs/lanes_example.yaml`) |
| `CONF_THRESHOLD` | `0.4` | Minimum detection confidence |
| `IOU_THRESHOLD` | `0.5` | NMS IoU threshold |
| `ALLOWED_CLASSES` | `car,truck,bus,motorcycle` | Comma-separated vehicle classes to count |
| `DB_PATH` | `counts.db` | SQLite database file path |

### Obtaining the RTSP URL

Most IP cameras expose an RTSP stream.  Common URL patterns:

```
rtsp://<user>:<password>@<ip>:<port>/stream        # generic
rtsp://<ip>:554/Streaming/Channels/101             # Hikvision main stream
rtsp://<ip>:554/cam/realmonitor?channel=1&subtype=0  # Dahua
rtsp://<ip>:554/h264/ch1/main/av_stream            # various
```

Check your camera manual or the [RTSP camera database](https://www.ispyconnect.com/cameras).

---

## Lane Calibration

1. Capture a single frame from the camera:

   ```bash
   python - <<'EOF'
   import cv2, os
   cap = cv2.VideoCapture(os.getenv("CAMERA_RTSP_URL", "rtsp://10.4.100.101:554/stream"))
   ret, frame = cap.read()
   if ret:
       cv2.imwrite("frame.jpg", frame)
       print(f"Saved frame.jpg ({frame.shape[1]}×{frame.shape[0]})")
   cap.release()
   EOF
   ```

2. Open `frame.jpg` in an image editor (e.g., GIMP, Paint) that shows pixel
   coordinates.

3. Identify the **two lane regions** and the **counting line** (typically a
   horizontal or diagonal line that vehicles must cross).

4. Edit `configs/lanes_example.yaml` (or create a new file) with your
   coordinates and set `LANE_CONFIG_PATH` in `.env`.

**Example** (960×720 camera, two horizontal lanes divided at y=360):

```yaml
direction: "any"
lane1_polygon:
  - [0,   0]
  - [960, 0]
  - [960, 360]
  - [0,   360]
lane2_polygon:
  - [0,   360]
  - [960, 360]
  - [960, 720]
  - [0,   720]
counting_line:
  - [0,   360]
  - [960, 360]
```

**Direction values:**

| Value | Meaning |
|---|---|
| `"any"` | Count crossings in both directions |
| `"positive"` | Count only downward crossings (increasing y) |
| `"negative"` | Count only upward crossings (decreasing y) |

---

## Network Access to the Camera

The surveillance camera has the private IP `10.4.100.101` and is on a
**different network** from the server running this application.  
Choose one of the following solutions:

### Option 1 – Site-to-Site VPN (recommended)

Set up **WireGuard** (or OpenVPN, IPsec) between the camera network and the
server network.  Once the VPN is up, the camera IP is directly routable.

```ini
# Example WireGuard config on the server side (/etc/wireguard/wg0.conf)
[Interface]
Address    = 10.200.0.1/24
PrivateKey = <server_private_key>
ListenPort = 51820

[Peer]
PublicKey  = <camera_router_public_key>
AllowedIPs = 10.4.100.0/24          # camera subnet
Endpoint   = <camera_router_public_ip>:51820
```

After `wg-quick up wg0`, the server can reach `10.4.100.101` directly.

### Option 2 – Static Route via a Gateway

If both networks are physically connected via a router/gateway that already
knows both subnets, add a static route on the server:

```bash
# Linux – add a persistent route to the camera subnet via a known gateway
ip route add 10.4.100.0/24 via <gateway_ip>
```

### Option 3 – NAT / Port-Forward (last resort)

Configure the router in the camera network to **port-forward** the RTSP port
(default 554) from its WAN interface to `10.4.100.101:554`.

```
# .env
CAMERA_RTSP_URL=rtsp://<router_wan_ip>:554/stream
```

> ⚠️ **Security warning:** Exposing an RTSP stream directly on the Internet is
> risky.  Use strong credentials, restrict by source IP with firewall rules,
> and prefer one of the VPN options above.

### Option 4 – RTSP Proxy in the Camera Network (MediaMTX)

Run [MediaMTX](https://github.com/bluenviron/mediamtx) on a small machine
(e.g., a Raspberry Pi) in the camera network.  MediaMTX re-publishes the
camera stream and can optionally add TLS/auth.

```yaml
# mediamtx.yml (minimal)
paths:
  cam1:
    source: rtsp://10.4.100.101:554/stream
```

Start with:

```bash
./mediamtx mediamtx.yml
```

Point the server to the proxy:

```
CAMERA_RTSP_URL=rtsp://<mediamtx_host>:8554/cam1
```

MediaMTX can expose the stream over WebRTC, HLS, or SRT in addition to RTSP,
giving you flexible access options without putting the camera directly on the
Internet.

---

## Docker

### Build and run

```bash
# Copy and edit the environment file
cp .env.example .env

# Build the image
docker build -t car-count:latest .

# Run (camera must be reachable from the container – see options below)
docker run -d \
  --name car_counter \
  -p 8000:8000 \
  --env-file .env \
  -v counts_data:/data \
  car-count:latest
```

### docker-compose

```bash
docker compose up -d
```

### Container network access to the camera

| Scenario | Solution |
|---|---|
| Camera reachable from the Docker **host** | `network_mode: host` in docker-compose.yml |
| WireGuard running on the host | `network_mode: host` or add the `wg0` interface to a custom Docker network |
| RTSP proxy running in another container | Put both containers on the same Docker network |
| Port-forward / public proxy | Default bridge networking; set `CAMERA_RTSP_URL` to the proxy address |

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | HTML dashboard (auto-refreshes every 3 s) |
| `GET` | `/api/stats` | JSON: counts, uptime, stream status, recent events |
| `GET` | `/api/frame` | Latest annotated JPEG frame (503 if none available) |

### `/api/stats` response example

```json
{
  "lane1": 42,
  "lane2": 37,
  "total": 79,
  "stream_status": "streaming",
  "last_update": "2026-02-24T13:00:00+00:00",
  "uptime_seconds": 3600.0,
  "recent_events": [
    {"ts": "2026-02-24T12:59:55+00:00", "lane": 1, "track_id": 101, "class_name": "car"},
    {"ts": "2026-02-24T12:59:57+00:00", "lane": 2, "track_id": 102, "class_name": "truck"}
  ]
}
```

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

The unit tests cover:

- **Point-in-polygon** lane assignment (6 cases)
- **Line-crossing detection** with direction filtering (6 cases)
- **Deduplication** – each track ID counted at most once (4 cases)

---

## GPU Acceleration

By default, inference runs on CPU.  To use a GPU:

1. Install the CUDA-enabled PyTorch **before** installing Ultralytics:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install ultralytics
   ```

2. Ultralytics automatically detects and uses a CUDA device if available.

3. In Docker, use the `nvidia/cuda` base image and pass `--gpus all` to
   `docker run`.

---

## Project Structure

```
carCount/
├── app/
│   ├── __init__.py
│   ├── config.py      # Pydantic settings + LaneConfig
│   ├── counter.py     # Point-in-polygon, line-crossing, LaneCounter
│   ├── database.py    # SQLite via aiosqlite
│   ├── detector.py    # YOLOv8 + ByteTrack wrapper
│   ├── main.py        # FastAPI routes + lifecycle
│   └── worker.py      # Background RTSP reader thread
├── configs/
│   └── lanes_example.yaml
├── tests/
│   ├── __init__.py
│   └── test_counter.py
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```
