"""Application configuration via environment variables and a YAML lane-config file."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Geometry helpers (plain data classes – no numpy dependency at config time)
# ---------------------------------------------------------------------------

Point = Tuple[float, float]
Polygon = List[Point]
Line = Tuple[Point, Point]


class LaneConfig:
    """Holds the parsed lane geometry."""

    def __init__(
        self,
        lane1_polygon: Polygon,
        lane2_polygon: Polygon,
        counting_line: Line,
        direction: str = "any",
    ) -> None:
        self.lane1_polygon: Polygon = lane1_polygon
        self.lane2_polygon: Polygon = lane2_polygon
        self.counting_line: Line = counting_line
        # "up"  → vehicle must be moving upward  (decreasing y) to be counted
        # "down" → vehicle must be moving downward (increasing y)
        # "any"  → count either direction
        self.direction: str = direction

    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str) -> "LaneConfig":
        data = cls._load(path)
        return cls(
            lane1_polygon=_parse_polygon(data["lane1_polygon"]),
            lane2_polygon=_parse_polygon(data["lane2_polygon"]),
            counting_line=_parse_line(data["counting_line"]),
            direction=data.get("direction", "any"),
        )

    @classmethod
    def _load(cls, path: str) -> dict:
        p = Path(path)
        text = p.read_text()
        if p.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(text)
        return json.loads(text)

    # default geometry (full-width, stacked lanes; counting line at y=360)
    @classmethod
    def default(cls) -> "LaneConfig":
        return cls(
            lane1_polygon=[(0, 0), (960, 0), (960, 360), (0, 360)],
            lane2_polygon=[(0, 360), (960, 360), (960, 720), (0, 720)],
            counting_line=((0, 360), (960, 360)),
            direction="any",
        )


def _parse_polygon(raw: list) -> Polygon:
    return [tuple(pt) for pt in raw]  # type: ignore[return-value]


def _parse_line(raw: list) -> Line:
    return (tuple(raw[0]), tuple(raw[1]))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# App settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    camera_rtsp_url: str = "rtsp://10.4.100.101:554/stream"
    model_path: str = "yolov8n.pt"
    lane_config_path: Optional[str] = None

    conf_threshold: float = 0.4
    iou_threshold: float = 0.5
    allowed_classes: List[str] = ["car", "truck", "bus", "motorcycle"]

    db_path: str = "counts.db"

    # How many recent events to keep in memory for /api/stats
    recent_events_limit: int = 100

    @field_validator("allowed_classes", mode="before")
    @classmethod
    def _split_classes(cls, v):
        if isinstance(v, str):
            return [c.strip() for c in v.split(",") if c.strip()]
        return v

    def get_lane_config(self) -> LaneConfig:
        if self.lane_config_path and os.path.isfile(self.lane_config_path):
            return LaneConfig.from_file(self.lane_config_path)
        return LaneConfig.default()


# Singleton – import this in other modules
settings = Settings()
