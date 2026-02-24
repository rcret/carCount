"""Background worker: reads RTSP stream, runs detection, updates counts."""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, Optional

import cv2
import numpy as np

from .config import settings
from .counter import LaneCounter
from .database import init_db, insert_event

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state (written by worker thread, read by API handlers)
# ---------------------------------------------------------------------------


class AppState:
    """Thread-safe shared state between the worker and the API."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.lane1_count: int = 0
        self.lane2_count: int = 0
        self.total_count: int = 0
        self.stream_status: str = "starting"
        self.last_update: Optional[str] = None
        self.latest_frame_bytes: Optional[bytes] = None
        self.start_time: float = time.time()
        self.recent_events: Deque[Dict] = deque(
            maxlen=settings.recent_events_limit
        )

    def update_counts(
        self,
        lane1: int,
        lane2: int,
        status: str,
        frame_bytes: Optional[bytes] = None,
    ) -> None:
        with self._lock:
            self.lane1_count = lane1
            self.lane2_count = lane2
            self.total_count = lane1 + lane2
            self.stream_status = status
            self.last_update = datetime.now(timezone.utc).isoformat()
            if frame_bytes is not None:
                self.latest_frame_bytes = frame_bytes

    def add_event(self, lane: int, track_id: int, class_name: str) -> None:
        with self._lock:
            self.recent_events.append(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "lane": lane,
                    "track_id": track_id,
                    "class_name": class_name,
                }
            )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "lane1": self.lane1_count,
                "lane2": self.lane2_count,
                "total": self.total_count,
                "stream_status": self.stream_status,
                "last_update": self.last_update,
                "uptime_seconds": round(time.time() - self.start_time, 1),
                "recent_events": list(self.recent_events),
            }


app_state = AppState()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _draw_annotations(
    frame: np.ndarray,
    detections,
    lane_config,
    lane1_count: int,
    lane2_count: int,
) -> np.ndarray:
    """Draw bounding boxes, lane polygons, counting line on *frame*."""
    annotated = frame.copy()

    # Draw lane polygons
    poly1 = np.array(lane_config.lane1_polygon, dtype=np.int32)
    poly2 = np.array(lane_config.lane2_polygon, dtype=np.int32)
    cv2.polylines(annotated, [poly1], True, (0, 255, 0), 2)
    cv2.polylines(annotated, [poly2], True, (255, 0, 0), 2)

    # Counting line
    (x1, y1), (x2, y2) = lane_config.counting_line
    cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    # Detections
    for x1b, y1b, x2b, y2b, tid, conf, cls in detections:
        cv2.rectangle(annotated, (x1b, y1b), (x2b, y2b), (255, 255, 0), 2)
        label = f"#{tid} {cls} {conf:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1b, max(y1b - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

    # Counts overlay
    cv2.putText(
        annotated,
        f"Lane1: {lane1_count}  Lane2: {lane2_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    return annotated


def _encode_jpeg(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


def _run_worker(loop: asyncio.AbstractEventLoop) -> None:
    """Worker function – runs in a daemon thread."""
    from .detector import Detector

    lane_config = settings.get_lane_config()
    counter = LaneCounter(lane_config)
    detector = Detector()

    rtsp_url = settings.camera_rtsp_url
    logger.info("Worker starting. RTSP URL: %s", rtsp_url)

    cap: Optional[cv2.VideoCapture] = None
    consecutive_failures = 0

    while True:
        if cap is None or not cap.isOpened():
            logger.info("Opening stream: %s", rtsp_url)
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                consecutive_failures += 1
                wait = min(30, 2**consecutive_failures)
                logger.warning(
                    "Cannot open stream, retry in %ds (attempt %d)",
                    wait,
                    consecutive_failures,
                )
                app_state.update_counts(
                    counter.lane1_count,
                    counter.lane2_count,
                    "disconnected",
                )
                time.sleep(wait)
                continue

        consecutive_failures = 0
        ret, frame = cap.read()
        if not ret:
            logger.warning("Frame read failed, re-opening stream.")
            cap.release()
            cap = None
            app_state.update_counts(
                counter.lane1_count,
                counter.lane2_count,
                "reconnecting",
            )
            time.sleep(1)
            continue

        try:
            detections = detector.detect_and_track(frame)
        except Exception as exc:  # noqa: BLE001
            logger.error("Detection error: %s", exc)
            detections = []

        for x1, y1, x2, y2, track_id, conf, class_name in detections:
            if track_id < 0:
                continue
            bx = (x1 + x2) / 2
            by = float(y2)
            new_lane = counter.update(track_id, (bx, by), class_name)
            if new_lane is not None:
                app_state.add_event(new_lane, track_id, class_name)
                # Persist asynchronously without blocking the worker thread
                asyncio.run_coroutine_threadsafe(
                    insert_event(new_lane, track_id, class_name), loop
                )

        annotated = _draw_annotations(
            frame, detections, lane_config, counter.lane1_count, counter.lane2_count
        )
        frame_bytes = _encode_jpeg(annotated)

        app_state.update_counts(
            counter.lane1_count,
            counter.lane2_count,
            "streaming",
            frame_bytes,
        )


def start_worker(loop: asyncio.AbstractEventLoop) -> threading.Thread:
    """Start the background worker thread and return it."""
    t = threading.Thread(target=_run_worker, args=(loop,), daemon=True, name="rtsp-worker")
    t.start()
    return t
