"""FastAPI application entry-point."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse

from .config import settings
from .database import init_db
from .worker import app_state, start_worker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await init_db()
    loop = asyncio.get_event_loop()
    start_worker(loop)
    logger.info("Application started. Worker thread launched.")
    yield


app = FastAPI(
    title="Vehicle Counter",
    description="Two-lane vehicle counter from an RTSP surveillance stream.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, summary="HTML dashboard")
async def dashboard() -> HTMLResponse:
    s = app_state.snapshot()
    status_color = {
        "streaming": "#2ecc71",
        "disconnected": "#e74c3c",
        "reconnecting": "#f39c12",
        "starting": "#3498db",
    }.get(s["stream_status"], "#95a5a6")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="3">
  <title>Vehicle Counter Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 20px; }}
    h1   {{ color: #e94560; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
    .card {{ background: #16213e; border-radius: 10px; padding: 20px; text-align: center; }}
    .card h2 {{ margin: 0 0 10px; font-size: 1rem; color: #aaa; }}
    .card p  {{ margin: 0; font-size: 2.5rem; font-weight: bold; color: #e94560; }}
    .status {{ display: inline-block; padding: 6px 16px; border-radius: 20px;
               background: {status_color}; color: #fff; font-weight: bold; }}
    .footer {{ margin-top: 30px; font-size: 0.8rem; color: #555; }}
    img {{ max-width: 100%; border-radius: 8px; margin-top: 20px; }}
  </style>
</head>
<body>
  <h1>🚗 Vehicle Counter</h1>
  <p>Stream status: <span class="status">{s['stream_status']}</span></p>
  <div class="grid">
    <div class="card"><h2>Lane 1</h2><p>{s['lane1']}</p></div>
    <div class="card"><h2>Lane 2</h2><p>{s['lane2']}</p></div>
    <div class="card"><h2>Total</h2><p>{s['total']}</p></div>
    <div class="card"><h2>Uptime</h2><p>{s['uptime_seconds']}s</p></div>
  </div>
  <p>Last update: {s['last_update'] or 'N/A'}</p>
  <p><a href="/api/frame" style="color:#e94560;">Latest annotated frame</a> &nbsp;|&nbsp;
     <a href="/api/stats" style="color:#e94560;">JSON stats</a></p>
  <div class="footer">Auto-refreshes every 3 s &bull; Camera: {settings.camera_rtsp_url}</div>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/api/stats", summary="JSON stats")
async def api_stats() -> JSONResponse:
    return JSONResponse(app_state.snapshot())


@app.get("/api/frame", summary="Latest annotated JPEG frame")
async def api_frame() -> Response:
    with app_state._lock:
        frame_bytes = app_state.latest_frame_bytes
    if frame_bytes is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No frame available yet. "
                "The worker may be connecting to the camera or the stream is unavailable."
            ),
        )
    return Response(content=frame_bytes, media_type="image/jpeg")
